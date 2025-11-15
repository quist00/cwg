import csv
import os
import sys
import subprocess
from pathlib import Path

# Configuration
default_batch_size = 15
data_dir = Path('backend/data')
csv_glob = 'pleco_top*.csv'
worksheet_prefix = 'worksheet_batch'


def find_latest_csv():
    files = sorted(data_dir.glob(csv_glob), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def read_fronts_from_csv(csv_path):
    fronts = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Accept both 'front' and 'Front' (case-insensitive)
        front_key = None
        for key in reader.fieldnames:
            if key.lower() == 'front':
                front_key = key
                break
        if not front_key:
            print('No "front" column found in CSV.')
            return []
        for row in reader:
            front = row.get(front_key, '').strip()
            if front:
                fronts.append(front)
    return fronts


def batch_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]


def generate_worksheet(batch, batch_num, guide=None):
    chars = ''.join(batch)
    pdf_name = f'{worksheet_prefix}_{batch_num:02d}.pdf'
    pdf_path = data_dir / pdf_name
    
    # gen.py requires --makemeahanzi and --cedict paths
    cmd = [
        sys.executable, 'cwg_gen.py',
        '--makemeahanzi', './makemeahanzi',
        '--cedict', './cedict',
        '--characters', chars
    ]
    if guide:
        cmd += ['--guide', guide]
    
    print(f'Generating worksheet: {pdf_path} ({len(batch)} items)')
    subprocess.run(cmd, check=True)
    
    # Move the generated sheet.pdf to the desired location
    generated_pdf = Path('sheet.pdf')
    if generated_pdf.exists():
        generated_pdf.rename(pdf_path)
    else:
        print(f'Warning: sheet.pdf not found after generation')


def main(batch_size=default_batch_size, guide=None):
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = find_latest_csv()
    if not csv_path:
        print('No CSV file found in backend/data.')
        sys.exit(1)
    fronts = read_fronts_from_csv(csv_path)
    if not fronts:
        print('No valid fronts found in CSV.')
        sys.exit(1)
    for idx, batch in enumerate(batch_list(fronts, batch_size), 1):
        generate_worksheet(batch, idx, guide=guide)
    print('All worksheets generated.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Batch generate worksheets from Pleco CSV output.')
    parser.add_argument('--batch-size', type=int, default=default_batch_size, help='Number of items per worksheet (default: 15)')
    parser.add_argument('--guide', type=str, default=None, help='Worksheet guide style (optional)')
    args = parser.parse_args()
    main(batch_size=args.batch_size, guide=args.guide)
