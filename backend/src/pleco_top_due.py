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
from datetime import datetime
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


def query_top_due(db_path: str, limit: int) -> List[Dict[str, Any]]:
    """
    Return a list of dicts:
        {'id': int, 'due': datetime|None, 'front': str, 'data': dict}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1) Try Anki-like schema first
    try:
        sql = (
            "SELECT c.id, c.due, n.sfld AS front, n.data "
            "FROM cards c JOIN notes n ON c.nid = n.id "
            "WHERE c.queue != -1 ORDER BY c.due ASC LIMIT ?"
        )
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
        model = "anki"
    except sqlite3.Error:
        rows = None
        model = None

    # 2) If not Anki, try Pleco schema
    if rows is None:
        # Find scores table like 'pleco_flash_scores_%' (choose the one with most rows)
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pleco_flash_scores_%'")
        score_tables = [r[0] for r in cur.fetchall()]
        if not score_tables:
            # Provide schema list for debugging
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tbls = [r[0] for r in cur.fetchall()]
            conn.close()
            raise RuntimeError(
                "Unsupported DB schema (no pleco scores table found). Tables: " + ", ".join(tbls)
            )

        # Pick the table with most rows
        table_counts = []
        for t in score_tables:
            try:
                cur.execute(f"SELECT COUNT(1) FROM {t}")
                cnt = cur.fetchone()[0]
            except sqlite3.Error:
                cnt = 0
            table_counts.append((cnt, t))
        table_counts.sort(reverse=True)
        scores_table = table_counts[0][1]

        # Inspect columns to guess due + card id
        def columns(table):
            cur.execute(f"PRAGMA table_info({table})")
            return [r[1] for r in cur.fetchall()]

        score_cols = columns(scores_table)

        # Candidate due column names (case-insensitive)
        due_candidates = [
            "nextreview", "next_review", "next", "due", "nextdue", "reviewtime", "nextr"
        ]
        due_col = next((c for c in score_cols if c.lower() in due_candidates), None)
        if due_col is None:
            # fallback: any column containing 'due' or 'next'
            due_col = next((c for c in score_cols if re.search(r"(due|next)", c, re.I)), None)
        if due_col is None:
            # Fallback: use lastreviewedtime as proxy for due (older => more due)
            if "lastreviewedtime" in (c.lower() for c in score_cols):
                due_col = next(c for c in score_cols if c.lower() == "lastreviewedtime")
            elif "firstreviewedtime" in (c.lower() for c in score_cols):
                due_col = next(c for c in score_cols if c.lower() == "firstreviewedtime")
            else:
                conn.close()
                raise RuntimeError(
                    f"Could not determine a due-like column in {scores_table}. Columns: {', '.join(score_cols)}"
                )

        # Candidate card id column names
        card_candidates = ["cardid", "card", "cid", "id"]
        card_col = next((c for c in score_cols if c.lower() in card_candidates), None)
        if card_col is None:
            # heuristic: first INTEGER primary key-like column
            cur.execute(f"PRAGMA table_info({scores_table})")
            rows_info = cur.fetchall()
            pk_cols = [r for r in rows_info if r[5] == 1]
            card_col = pk_cols[0][1] if pk_cols else score_cols[0]

        # Fetch top due cards from scores table
        # Order so that rows with a real timestamp come first and are earliest due
        cur.execute(
            (
                f"SELECT {card_col} AS card_id, {due_col} AS due FROM {scores_table} "
                f"ORDER BY CASE WHEN {due_col} IS NULL OR {due_col}<=0 THEN 1 ELSE 0 END, {due_col} ASC LIMIT ?"
            ),
            (limit,)
        )
        score_rows = cur.fetchall()

        # Try to resolve front text from pleco_flash_cards
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pleco_flash_cards'")
        has_cards = cur.fetchone() is not None
        fronts = {}
        if has_cards and score_rows:
            cards_cols = columns("pleco_flash_cards")
            # plausible text columns, choose the first available
            text_candidates = [
                "text", "head", "hw", "word", "entry", "front", "question", "chars", "hanzi", "headword"
            ]
            text_col = next((c for c in cards_cols if c.lower() in text_candidates), None)

            # find id column in cards table
            id_candidates = ["id", "cardid", "cid", "card"]
            id_col = next((c for c in cards_cols if c.lower() in id_candidates), None)

            if text_col and id_col:
                ids = tuple({r["card_id"] for r in score_rows})
                placeholder = ",".join(["?"] * len(ids)) if ids else "?"
                cur.execute(
                    f"SELECT {id_col} AS id, {text_col} AS front FROM pleco_flash_cards WHERE {id_col} IN (" + placeholder + ")",
                    tuple(ids) if ids else (None,)
                )
                for rr in cur.fetchall():
                    fronts[rr["id"]] = rr["front"]

        # Build unified rows structure like the Anki branch
        rows = []
        for r in score_rows:
            rows.append({
                "id": r["card_id"],
                "due": r["due"],
                "front": fronts.get(r["card_id"], ""),
                "data": {}
            })
        model = "pleco"

    conn.close()

    result = []
    for r in rows:
        due_val = r["due"]
        # Convert due to datetime; detect unit (ms vs sec)
        due_dt = None
        if isinstance(due_val, (int, float)) and due_val > 0:
            # Heuristic: >= 10^12 implies ms
            if due_val >= 10**12:
                due_dt = datetime.fromtimestamp(due_val / 1000.0)
            else:
                due_dt = datetime.fromtimestamp(due_val)

        data_json = {}
        if "data" in r.keys():
            try:
                data_json = json.loads(r["data"]) if isinstance(r["data"], str) else {}
            except json.JSONDecodeError:
                data_json = {}

        result.append({
            "id": r["id"],
            "due": due_dt,
            "front": r.get("front", ""),
            "data": data_json,
        })
    # Ensure dict keys include 'front' and a sortable 'due_sort'
    for r in result:
        if 'front' not in r:
            r['front'] = ''
        # unify due field
        if 'due_sort' not in r and 'due' in r:
            r['due_sort'] = r['due']
    return result


def pretty_print(cards):
    print("\nTop due cards".center(60, "="))
    header = f"{'#':>3}  {'Due':19}  {'Front'}"
    print(header)
    print("-" * len(header))
    for i, c in enumerate(cards, 1):
        due_str = c["due"].strftime("%Y-%m-%d %H:%M") if c["due"] else "New"
        print(f"{i:>3}  {due_str:19}  {_clean_front(c['front'])}")
    print("=" * 60)


def write_csv(cards: List[Dict[str, Any]], out_path: Path) -> None:
    """CSV with just the fields you probably need."""
    import csv
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['#', 'Front', 'Due'])
        for idx, c in enumerate(cards, start=1):
            front = _clean_front(c.get('front', ''))
            due = c.get('due_sort') or c.get('due') or ''
            w.writerow([idx, front, due])
    print(f"CSV written to: {out_path}")


# Optional helper to print concatenated string for worksheet generator
def print_compound_sequence(cards: List[Dict[str, Any]]) -> None:
    seq = ''.join(_clean_front(c.get('front', '')) for c in cards)
    print(f"\nWorksheet sequence:\n{seq}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract top N due Pleco cards")
    parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=20,
        help="How many cards to pull (default: 20)",
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

        # ------------------------------------------------------------------- #
        # 3. Query
        # ------------------------------------------------------------------- #
        cards = query_top_due(db_path, args.limit)

        # ------------------------------------------------------------------- #
        # 4. Output
        # ------------------------------------------------------------------- #
        pretty_print(cards)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path.cwd() / f"pleco_top{args.limit}_{timestamp}.csv"
        write_csv(cards, csv_path)

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
            words = [c["front"] for c in cards]
            import subprocess

            cmd = [sys.executable, str(cli_path)] + words
            print(f"\nExecuting your CLI: {' '.join(cmd)}")
            subprocess.run(cmd, check=False)

    finally:
        # Clean up the temporary extraction directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()