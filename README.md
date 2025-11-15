# Chinese Worksheet Generator
Allows one to generate Chinese practice worksheets.

[![Build Status](https://travis-ci.org/lucivpav/cwg.svg?branch=master)](https://travis-ci.org/lucivpav/cwg)

![](http://i.imgur.com/HH9eKtC.png)

## Features
* Simplified and traditional Chinese
* Stroke order
* Radicals
* Words
* Customizable pinyin and translation
* Customizable title and grid style

## Dependencies
* [Make Me a Hanzi dataset](https://github.com/skishore/makemeahanzi)
* [CEDICT dataset](https://www.mdbg.net/chinese/dictionary?page=cedict)
* cairosvg
* reportlab
* flask
* [SourceHanSansTC-Normal.ttf](https://github.com/be5invis/source-han-sans-ttf/releases)
* [TagManager](https://maxfavilli.com/jquery-tag-manager)

## Installation notes
* Place TagManager folder into *frontend* folder
* [Windows 10 64-bit notes](https://github.com/lucivpav/cwg/wiki/Windows-10-64-bit-installation-notes)

## Windows Quickstart
- Install Python 3.10+ (64-bit). Optionally install Pipenv (`pip install pipenv`).
- From a PowerShell window in the repo root:

```powershell
# 1) Fetch datasets/assets (makemeahanzi, cedict, TagManager, font)
powershell -ExecutionPolicy Bypass -File .\setup.ps1

# 2) Create env and install Python deps (choose one)

# Option A: Pipenv
pip install pipenv
pipenv install

# Option B: venv + pip
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install cairosvg reportlab flask flask-restful flask-jsonpify flask-cors pytest pytest-mock pytest-cov
```

If `SourceHanSansTC-Normal.ttf` is not created automatically, install 7‑Zip and re-run `setup.ps1`, or manually place the font file in the repo root.

If Cairo DLL error occurs on import (Windows only):
- Install MSYS2 (https://www.msys2.org/), then install Cairo (prefer UCRT64):
	- In the MSYS2 UCRT64 shell:
		- `pacman -Syu` (restart shell if prompted), then `pacman -S mingw-w64-ucrt-x86_64-cairo`
- Add the bin directory to PATH in PowerShell and persist for future sessions:
```powershell
$CairoPath = 'C:\\msys64\\ucrt64\\bin'    # or 'C:\\msys64\\mingw64\\bin' if you used MINGW64
$env:PATH = "$CairoPath;$env:PATH"
setx PATH "$CairoPath;$([Environment]::GetEnvironmentVariable('PATH','User'))"

# Ensure your venv stays first in PATH (reactivate if needed)
. .\.venv\Scripts\Activate.ps1

# Verify DLL is visible and import works
@"
import ctypes
ctypes.cdll.LoadLibrary('libcairo-2.dll')
print('Cairo DLL OK')
"@ | python -
python -c "import cairosvg; print('CairoSVG OK')"
```

## Words
* Use parentheses to group multiple characters together. This will add definition of such words into the sheet.

## Command line worksheet generation
### Show usage
```
python cwg_gen.py
```
### Generate worksheet
```
python cwg_gen.py --makemeahanzi .\makemeahanzi --cedict .\cedict --characters '你好' --title 'Vocabulary' --guide star --stroke-order-color red
```
### Customize pinyin, translation and words
```
python cwg_gen.py --makemeahanzi .\makemeahanzi --cedict .\cedict --characters '(你好)' --info   # Generate character_infos.json

# You may edit the 'character_infos.json' and 'word_infos.json' to customize pinyin, translation and words

python cwg_gen.py --makemeahanzi .\makemeahanzi --title 'Vocabulary' --guide star --sheet   # Generate worksheet
```

### `--guide` values
- `none`: no marks (default)
- `star`: diagonal X
- `cross`: plus sign
- `cross_star`: cross + star
- `character`: faint character in practice squares

## Pleco Integration (Automated Workflow)
### Extract prioritized flashcards from Pleco backup
```powershell
python backend\src\pleco_top_due.py
```
This extracts the top 140 most-due cards from your latest Pleco backup (`.pqb` file) and saves them to `backend/data/pleco_top140_TIMESTAMP.csv`. The script uses an advanced SRS priority algorithm to rank cards based on difficulty, accuracy, overdue ratio, and volatility.

### Generate batch worksheets from Pleco CSV
```powershell
python batch_generate_worksheets.py
```
This reads the latest CSV output from `pleco_top_due.py`, batches the characters into groups of 15 (configurable with `--batch-size`), and generates worksheet PDFs saved to your iCloud Drive (`D:\DaveApple\files\iCloudDrive\Mandarin\worksheets`). By default, worksheets use `cross_star` guide style and `red` stroke order color.

Options:
- `--batch-size N`: Number of items per worksheet (default: 15)
- `--guide STYLE`: Worksheet guide style (default: cross_star; options: none, star, cross, cross_star, character)
- `--stroke-order-color COLOR`: Stroke order color (default: red)

Example:
```powershell
python batch_generate_worksheets.py --batch-size 20 --guide star --stroke-order-color black
```

## Running tests
```
pipenv install
cd backend
pipenv run pytest test
```

## Troubleshooting (Windows)
- Scripts blocked: run with policy bypass
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```
- Cairo DLL not found: install MSYS2 Cairo and add `C:\msys64\ucrt64\bin` (or `mingw64\bin`) to PATH, then reactivate your venv.
- Wrong Python on PATH: ensure `.venv\Scripts` is first (reactivate venv) and keep MSYS2 bin later in PATH.
- Missing font: ensure `SourceHanSansTC-Normal.ttf` is in the repo root; `setup.ps1` can fetch it if 7‑Zip is installed.

## License
This project is released under the GPLv3 license, for more details, take a look at the LICENSE.txt file in the source code.
