#!/usr/bin/env python
"""
pleco_top_due.py
----------------
Find the newest Pleco backup, extract its SQLite DB and return the N most-due
flashcards.  The list of front-field texts is printed, saved to CSV and
optionally handed to another Python CLI you already have.

Usage
-----
    python pleco_top_due.py            # default N = 20
    python pleco_top_due.py 50         # pull 50 cards
    python pleco_top_due.py 10 --run   # pull 10 and then exec your tool
"""

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# CONFIGURATION
#
# By default we search under your iCloud Drive sync root for any directory
# named "Flashcard Backups" and pick the most recent *.pqb inside.
# You can override via:
#   - env var PLECO_BACKUP_ROOT (points either to the base iCloud path or
#     directly to the "Flashcard Backups" directory), or
#   - CLI flag --root PATH
# --------------------------------------------------------------------------- #
DEFAULT_BASE = Path(r"D:\DaveApple\files\iCloudDrive")
ENV_ROOT = os.environ.get("PLECO_BACKUP_ROOT")
BACKUP_ROOT = Path(ENV_ROOT) if ENV_ROOT else DEFAULT_BASE
# --------------------------------------------------------------------------- #


def _collect_pqb_in_dir(folder: Path):
    return [p for p in folder.glob("*.pqb") if p.is_file()]


def find_latest_pqb(search_root: Path) -> Path:
    """Return the Path of the most-recent *.pqb found starting at search_root.

    - If search_root is a folder named exactly "Flashcard Backups", search only there.
    - Otherwise, search recursively for all folders named "Flashcard Backups" beneath it.
    """
    candidates = []
    try:
        if search_root.name.lower() == "flashcard backups":
            candidates.extend(_collect_pqb_in_dir(search_root))
        else:
            # Search recursively for any "Flashcard Backups" directories and gather PQB files
            for d in search_root.rglob("Flashcard Backups"):
                if d.is_dir():
                    candidates.extend(_collect_pqb_in_dir(d))
    except FileNotFoundError:
        pass

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No .pqb files found under {search_root}. If your backups are elsewhere, "
            f"set PLECO_BACKUP_ROOT or pass --root to point to either the base iCloud "
            f"path or directly to the 'Flashcard Backups' folder."
        )
    return candidates[0]


def _is_sqlite(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
        return header == b"SQLite format 3\x00"
    except Exception:
        return False


def extract_db(pqb_path: Path, temp_dir: Path) -> Path:
    """
    Return a path to a SQLite DB extracted from the backup.

    Supports:
    - .pqb as a ZIP containing flashcards.db (preferred)
    - .pqb as a bare SQLite file (some Pleco exports)
    """
    # Case 1: PQB is a ZIP archive
    if zipfile.is_zipfile(pqb_path):
        with zipfile.ZipFile(pqb_path, "r") as z:
            # Prefer an exact flashcards.db, else first .db inside the archive
            db_member = None
            if "flashcards.db" in z.namelist():
                db_member = "flashcards.db"
            else:
                db_candidates = [n for n in z.namelist() if n.lower().endswith(".db")]
                if db_candidates:
                    db_member = db_candidates[0]
            if not db_member:
                raise RuntimeError("No SQLite .db file found inside the backup archive")

            extracted = temp_dir / "flashcards.db"
            with z.open(db_member) as src, open(extracted, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return extracted

    # Case 2: PQB is itself a SQLite file
    if _is_sqlite(pqb_path):
        extracted = temp_dir / "flashcards.db"
        shutil.copyfile(pqb_path, extracted)
        return extracted

    # Otherwise, likely a cloud placeholder or unsupported format
    size = pqb_path.stat().st_size if pqb_path.exists() else 0
    raise RuntimeError(
        "Backup file is not a ZIP or SQLite. If this is an iCloud placeholder, "
        "mark it as 'Always keep on this device' in Explorer so the full file downloads. "
        f"Path={pqb_path}, size={size} bytes"
    )


def _clean_front(text: str) -> str:
    if not text:
        return ""
    # Remove ASCII and fullwidth at signs plus incidental spaces
    cleaned = (text
               .replace('@', '')
               .replace('＠', '')
               .replace(' ', '')
               )
    # Wrap multi‑char strings in parentheses (avoid double)
    if len(cleaned) > 1 and not (cleaned.startswith('(') and cleaned.endswith(')')):
        cleaned = f'({cleaned})'
    return cleaned


def _ts_to_dt(ts: int | float | None) -> datetime | None:
    """Convert a Pleco timestamp (seconds or ms) to datetime."""
    if not isinstance(ts, (int, float)) or ts <= 0:
        return None
    # Heuristic: milliseconds if very large
    if ts >= 10**12:
        ts /= 1000.0
    try:
        return datetime.fromtimestamp(ts)
    except OSError:
        return None


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _compute_predicted_interval_hours(row: dict) -> float:
    """
    Derive a 'predicted interval' (hours) using:
      - base interval from (last - first)/(reviews - 1)
      - growth factor from score, difficulty, accuracy
    Falls back to modest default if insufficient history.
    """
    first_dt = _ts_to_dt(row.get("firstreviewedtime"))
    last_dt = _ts_to_dt(row.get("lastreviewedtime"))
    reviewed = row.get("reviewed", 0) or 0
    score = row.get("score", 0) or 0
    difficulty = row.get("difficulty", 0) or 0
    correct = row.get("correct", 0) or 0
    incorrect = row.get("incorrect", 0) or 0
    accuracy = _safe_div(correct, (correct + incorrect)) if (correct + incorrect) > 0 else 0.75

    # Base interval (hours)
    if first_dt and last_dt and reviewed and reviewed > 1 and last_dt > first_dt:
        total_span_hours = (last_dt - first_dt).total_seconds() / 3600.0
        base_interval = total_span_hours / (reviewed - 1)
        # Clamp unreasonable low/high
        base_interval = max(0.25, min(base_interval, 24 * 30))
    else:
        # Fallback base proportional to score
        base_interval = 4.0 + (score / 100.0) * 12.0  # 4h .. ~16h typical

    # Growth factors
    score_factor = 1.0 + min(score, 300) / 300.0          # up to 2.0
    difficulty_factor = 1.0 / (1.0 + 0.7 * max(difficulty, 0))  # harder -> shorter
    accuracy_factor = max(0.6, min(accuracy * 1.1, 1.25))       # 0.6 .. 1.25

    predicted = base_interval * score_factor * difficulty_factor * accuracy_factor
    # Global clamps
    predicted = max(0.25, min(predicted, 24 * 60))  # 15 min .. 60 days
    return predicted


def _compute_priority(row: dict) -> dict:
    """
    Compute spaced-repetition priority.
    Returns dict of metrics appended to row:
      priority, predicted_interval_hours, elapsed_hours, overdue_ratio, accuracy, error_rate, volatility
    """
    last_dt = _ts_to_dt(row.get("lastreviewedtime"))
    first_dt = _ts_to_dt(row.get("firstreviewedtime"))
    now = datetime.now()

    score = row.get("score", 0) or 0
    difficulty = row.get("difficulty", 0) or 0
    correct = row.get("correct", 0) or 0
    incorrect = row.get("incorrect", 0) or 0
    reviewed = row.get("reviewed", 0) or 0
    scoreinctime = row.get("scoreinctime")
    scoredectime = row.get("scoredectime")

    accuracy = _safe_div(correct, (correct + incorrect)) if (correct + incorrect) > 0 else 0.75
    error_rate = _safe_div(incorrect, (correct + incorrect)) if (correct + incorrect) > 0 else 0.0

    predicted_hours = _compute_predicted_interval_hours(row)

    elapsed_hours = _safe_div((now - last_dt).total_seconds(), 3600.0) if last_dt else 0.0
    overdue_ratio = max(0.0, _safe_div((elapsed_hours - predicted_hours), predicted_hours))

    # Volatility heuristic
    v = 0.2
    if scoredectime and scoreinctime and scoredectime > scoreinctime:
        v = 1.0
    elif scoreinctime and (not scoredectime or scoreinctime > scoredectime):
        # recent increase => medium volatility
        v = 0.5

    # Recency penalty for very new cards
    recency_penalty = 0.0
    if first_dt:
        age_hours = _safe_div((now - first_dt).total_seconds(), 3600.0)
        if age_hours < 12 and reviewed < 3:
            recency_penalty = (12 - age_hours) / 12.0  # fades to 0 over first 12h

    # Normalized difficulty
    difficulty_norm = difficulty / (difficulty + 3.0) if difficulty >= 0 else 0.0

    # Weight tuning (initial)
    w1, w2, w3, w4, w5, w6 = 0.45, 0.15, 0.15, 0.10, 0.10, 0.05
    priority = (
        w1 * overdue_ratio +
        w2 * v +
        w3 * error_rate +
        w4 * difficulty_norm +
        w5 * (1.0 - accuracy) +
        w6 * recency_penalty
    )

    # Add metrics to row
    row["priority"] = round(priority, 6)
    row["predicted_interval_hours"] = round(predicted_hours, 3)
    row["elapsed_hours"] = round(elapsed_hours, 3)
    row["overdue_ratio"] = round(overdue_ratio, 3)
    row["accuracy"] = round(accuracy, 3)
    row["error_rate"] = round(error_rate, 3)
    row["volatility"] = v
    return row


def query_top_due(db_path: str, limit: int) -> List[Dict[str, Any]]:
    """
    Advanced version:
      - For Pleco: derive priority from score/difficulty/timing metrics.
      - For Anki-like: still uses c.due if available, but also computes priority via elapsed/score proxy.
      Returns list of rows with added SRS metrics.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = []
    model = None

    # Try Anki-like first
    try:
        cur.execute(
            "SELECT c.id, c.due, n.sfld AS front, n.data, 0 AS score, 0 AS difficulty, 0 AS correct, 0 AS incorrect, 0 AS reviewed, 0 AS firstreviewedtime, c.due AS lastreviewedtime, 0 AS scoreinctime, 0 AS scoredectime "
            "FROM cards c JOIN notes n ON c.nid = n.id "
            "WHERE c.queue != -1 LIMIT ?", (limit * 5,)
        )
        rows = [dict(r) for r in cur.fetchall()]
        if rows:
            model = "anki"
    except sqlite3.Error:
        rows = []

    if not rows:
        # Pleco path
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pleco_flash_scores_%'")
        score_tables = [r[0] for r in cur.fetchall()]
        if not score_tables:
            conn.close()
            raise RuntimeError("No Pleco scores table found.")

        # Choose largest
        best_table = None
        best_count = -1
        for t in score_tables:
            try:
                cur.execute(f"SELECT COUNT(1) FROM {t}")
                cnt = cur.fetchone()[0]
                if cnt > best_count:
                    best_count = cnt
                    best_table = t
            except sqlite3.Error:
                pass

        if not best_table:
            conn.close()
            raise RuntimeError("Unable to select Pleco scores table.")

        # Load raw score rows
        cur.execute(f"PRAGMA table_info({best_table})")
        score_cols = [c[1] for c in cur.fetchall()]

        # Card id column
        id_col = next((c for c in score_cols if c.lower() in ("cardid", "card", "cid", "id")), score_cols[0])

        cur.execute(f"SELECT * FROM {best_table}")
        raw_scores = [dict(r) for r in cur.fetchall()]

        # Load front texts
        fronts = {}
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pleco_flash_cards'")
        if cur.fetchone():
            cur.execute("PRAGMA table_info(pleco_flash_cards)")
            card_cols = [c[1] for c in cur.fetchall()]
            text_col = next((c for c in card_cols if c.lower() in (
                "text", "head", "hw", "word", "entry", "front", "question", "chars", "hanzi", "headword")), None)
            id2_col = next((c for c in card_cols if c.lower() in ("id", "cardid", "cid", "card")), None)
            if text_col and id2_col:
                ids = tuple({row[id_col] for row in raw_scores})
                if ids:
                    placeholders = ",".join(["?"] * len(ids))
                    cur.execute(f"SELECT {id2_col} AS id, {text_col} AS front FROM pleco_flash_cards WHERE {id2_col} IN ({placeholders})", ids)
                    for rr in cur.fetchall():
                        fronts[rr["id"]] = rr["front"]

        # Build uniform rows
        for r in raw_scores:
            rid = r.get(id_col)
            r_out = {
                "id": rid,
                "front": fronts.get(rid, ""),
                # Carry raw timing/performance fields
                "score": r.get("score", 0),
                "difficulty": r.get("difficulty", 0),
                "correct": r.get("correct", 0),
                "incorrect": r.get("incorrect", 0),
                "reviewed": r.get("reviewed", 0),
                "firstreviewedtime": r.get("firstreviewedtime", 0),
                "lastreviewedtime": r.get("lastreviewedtime", 0),
                "scoreinctime": r.get("scoreinctime", 0),
                "scoredectime": r.get("scoredectime", 0),
            }
            rows.append(r_out)
        model = "pleco"

    conn.close()

    # Compute metrics / priority
    enriched = []
    now = datetime.now()
    for r in rows:
        # For Anki: treat 'due' original as numeric; convert lastreviewed fallback
        if model == "anki":
            due_raw = r.get("due")
            due_dt = None
            if isinstance(due_raw, (int, float)) and due_raw > 0:
                if due_raw >= 10**12:
                    due_dt = datetime.fromtimestamp(due_raw / 1000.0)
                else:
                    due_dt = datetime.fromtimestamp(due_raw)
            r["lastreviewedtime"] = due_raw  # approximate
        enriched.append(_compute_priority(r))

    # Sort by priority (desc), tie-break by elapsed_hours then lower score
    enriched.sort(key=lambda x: (x["priority"],
                                 x.get("elapsed_hours", 0),
                                 -x.get("score", 0)), reverse=True)

    # Limit
    top = enriched[:limit]

    # Compute a synthetic next-review datetime (predicted due) for display
    result = []
    for r in top:
        last_dt = _ts_to_dt(r.get("lastreviewedtime"))
        predicted_interval_hours = r["predicted_interval_hours"]
        due_dt = last_dt + timedelta(hours=predicted_interval_hours) if last_dt else None
        result.append({
            "id": r["id"],
            "front": r.get("front", ""),
            "due": due_dt,
            "due_sort": due_dt,
            # Expose metrics for CSV/log if desired
            "priority": r["priority"],
            "predicted_interval_hours": r["predicted_interval_hours"],
            "elapsed_hours": r["elapsed_hours"],
            "overdue_ratio": r["overdue_ratio"],
            "accuracy": r["accuracy"],
            "difficulty": r.get("difficulty", 0),
            "score": r.get("score", 0),
        })
    return result


# Adjust pretty_print to show priority and predicted interval
def pretty_print(cards):
    print("\nTop due cards (advanced SRS)".center(85, "="))
    header = f"{'#':>3}  {'Next (est)':19}  {'Pri':>6}  {'PredInt(h)':>9}  {'Acc':>5}  {'Front'}"
    print(header)
    print("-" * len(header))
    for i, c in enumerate(cards, 1):
        due_str = c["due"].strftime("%Y-%m-%d %H:%M") if c["due"] else "—"
        print(f"{i:>3}  {due_str:19}  {c['priority']:6.3f}  {c['predicted_interval_hours']:9.2f}  {c['accuracy']:5.2f}  {_clean_front(c['front'])}")
    print("=" * 85)


def write_csv(cards: List[Dict[str, Any]], out_path: Path) -> None:
    """CSV (skips missing fronts). Includes priority + metrics."""
    kept = 0
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['#', 'Front', 'NextReview', 'Priority', 'PredIntervalHours', 'ElapsedHours', 'Accuracy', 'Score', 'Difficulty'])
        for idx, c in enumerate(cards, start=1):
            raw_front = c.get('front', '') or ''
            front = _clean_front(raw_front)
            if not raw_front or not front:
                continue
            due = c.get('due')
            due_str = due.strftime("%Y-%m-%d %H:%M") if isinstance(due, datetime) else ''
            w.writerow([
                idx,
                front,
                due_str,
                c.get('priority', ''),
                c.get('predicted_interval_hours', ''),
                c.get('elapsed_hours', ''),
                c.get('accuracy', ''),
                c.get('score', ''),
                c.get('difficulty', ''),
            ])
            kept += 1
    print(f"CSV written to: {out_path} (rows kept: {kept})")


# Optional helper to print concatenated string for worksheet generator
def print_compound_sequence(cards: List[Dict[str, Any]]) -> None:
    seq_parts = []
    for c in cards:
        raw_front = c.get('front', '') or ''
        cleaned = _clean_front(raw_front)
        if not raw_front or not cleaned:
            continue
        seq_parts.append(cleaned)
    seq = ''.join(seq_parts)
    print(f"\nWorksheet sequence (missing fronts omitted):\n{seq}\n")


def log_run(log_path: Path,
            backup_file: Path,
            db_path: Path,
            cards: List[Dict[str, Any]],
            missing_ids: List[int],
            csv_path: Path,
            sequence: str) -> None:
    lines = []
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append(f"Backup file: {backup_file}")
    lines.append(f"DB path: {db_path}")
    lines.append(f"Total cards queried: {len(cards)}")
    lines.append(f"Missing front count: {len(missing_ids)}")
    if missing_ids:
        lines.append("Missing card IDs: " + ', '.join(map(str, missing_ids)))
    lines.append(f"CSV path: {csv_path}")
    lines.append(f"Worksheet sequence (truncated to 500 chars): {sequence[:500]}")
    log_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Log written: {log_path}")


def inspect_schema(db_path: Path, sample_rows: int = 0) -> None:
    """Print SQLite tables, columns, row counts, and optional samples."""
    def _quote_ident(name: str) -> str:
        # Double up embedded " inside identifiers per SQL rules and wrap in "
        return '"' + str(name).replace('"', '""') + '"'

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("select sqlite_version()")
    ver = cur.fetchone()[0]
    print(f"\nSQLite version: {ver}")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    if not tables:
        print("No tables found.")
        conn.close()
        return

    for t in tables:
        print(f"\nTable: {t}")
        qt = _quote_ident(t)

        # Columns
        try:
            cur.execute(f"PRAGMA table_info({qt})")
            cols = cur.fetchall()
            if cols:
                print("  Columns:")
                for cid, name, ctype, notnull, dflt, pk in cols:
                    nn = " NOT NULL" if notnull else ""
                    pkflag = " PK" if pk else ""
                    dflt_str = f" DEFAULT {dflt}" if dflt is not None else ""
                    print(f"    - {name} {ctype}{nn}{pkflag}{dflt_str}")
            else:
                print("  Columns: (none)")
        except sqlite3.Error as e:
            print(f"  (Error reading columns: {e})")

        # Count
        try:
            cur.execute(f"SELECT COUNT(1) FROM {qt}")
            cnt = cur.fetchone()[0]
            print(f"  Rows: {cnt}")
        except sqlite3.Error as e:
            print(f"  (Error counting rows: {e})")
            cnt = 0

        # Sample rows
        if sample_rows and cnt:
            try:
                cur.execute(f"PRAGMA table_info({qt})")
                colnames = [r[1] for r in cur.fetchall()]
                cur.execute(f"SELECT * FROM {qt} LIMIT ?", (sample_rows,))
                rows = cur.fetchall()
                print("  Sample:")
                for r in rows:
                    preview = {colnames[i]: r[i] for i in range(len(colnames))}
                    print(f"    {preview}")
            except sqlite3.Error as e:
                print(f"  (Error sampling rows: {e})")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Extract top N due Pleco cards")
    parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=140,   # was 20
        help="How many cards to pull (default: 140)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="After printing, exec your own CLI with the front words as args",
    )
    parser.add_argument(
        "--cli",
        default=None,
        help="Path to your existing .py CLI (only needed with --run)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Override the backup search root. Point either to your iCloud base "
            "(e.g. D\\DaveApple\\files\\iCloudDrive) or directly to the 'Flashcard Backups' folder."
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print all tables, columns, and row counts, then exit",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="With --inspect, also print up to N sample rows per table",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------- #
    # 1. Find latest backup
    # ------------------------------------------------------------------- #
    search_root = Path(args.root) if args.root else BACKUP_ROOT
    latest_pqb = find_latest_pqb(search_root)
    print(f"Latest backup: {latest_pqb.name}  (modified {datetime.fromtimestamp(latest_pqb.stat().st_mtime)})")

    # ------------------------------------------------------------------- #
    # 2. Extract DB into a temp folder
    # ------------------------------------------------------------------- #
    temp_dir = Path(tempfile.mkdtemp(prefix="pleco_"))
    try:
        db_path = extract_db(latest_pqb, temp_dir)
        print(f"Extracted DB → {db_path}")

        # Inspect-only mode
        if args.inspect:
            inspect_schema(db_path, sample_rows=args.sample)
            return

        # ------------------------------------------------------------------- #
        # 3. Query
        # ------------------------------------------------------------------- #
        cards = query_top_due(db_path, args.limit)

        # Identify missing fronts
        missing_ids = [c["id"] for c in cards if not (c.get("front") or "").strip()]

        # Output (pretty_print keeps them visible; sanitized sequence skips them)
        pretty_print(cards)
        print(f"\nMissing fronts ({len(missing_ids)}): {missing_ids if missing_ids else 'None'}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path.cwd() / f"pleco_top{args.limit}_{timestamp}.csv"
        write_csv(cards, csv_path)

        # Build worksheet sequence excluding missing
        seq_parts = []
        for c in cards:
            raw_front = c.get('front', '')
            cleaned = _clean_front(raw_front)
            if raw_front and cleaned:
                seq_parts.append(cleaned)
        sequence = ''.join(seq_parts)
        print(f"\nWorksheet sequence (for generator):\n{sequence}\n")

        # Log file
        log_path = Path.cwd() / f"pleco_top_due_log_{timestamp}.txt"
        log_run(log_path, latest_pqb, db_path, cards, missing_ids, csv_path, sequence)

        # ------------------------------------------------------------------- #
        # 5. OPTIONAL: hand the words to your own CLI
        # ------------------------------------------------------------------- #
        if args.run:
            if not args.cli:
                print("--run requires --cli PATH/TO/your_script.py", file=sys.stderr)
                sys.exit(1)

            cli_path = Path(args.cli)
            if not cli_path.is_file():
                print(f"CLI not found: {cli_path}", file=sys.stderr)
                sys.exit(1)

            # Build argument list: your_script.py word1 word2 word3 ...
            words = [ _clean_front(c["front"]) for c in cards if c.get("front") and _clean_front(c["front"]) ]
            import subprocess

            cmd = [sys.executable, str(cli_path)] + words
            print(f"\nExecuting your CLI: {' '.join(cmd)}")
            subprocess.run(cmd, check=False)

    finally:
        # Clean up the temporary extraction directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()