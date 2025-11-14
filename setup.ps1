Param()

# Windows setup script mirroring setup.sh
# - Clones makemeahanzi
# - Downloads CEDICT data
# - Downloads TagManager assets
# - Attempts to fetch SourceHanSansTC-Normal.ttf (requires 7-Zip)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Ensure-EmptyDir($Path) {
    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

Write-Host 'Setting up datasets and assets...' -ForegroundColor Cyan

# Paths
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $RepoRoot

try {
    # Clean previous
    if (Test-Path 'cedict') { Remove-Item -Recurse -Force 'cedict' }
    if (Test-Path 'makemeahanzi') { Remove-Item -Recurse -Force 'makemeahanzi' }
    if (Test-Path 'frontend/tagmanager') { Remove-Item -Recurse -Force 'frontend/tagmanager' }

    # Clone makemeahanzi
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Warning 'git not found. Please install Git and re-run.'
        throw 'Missing dependency: git'
    }
    git clone https://github.com/skishore/makemeahanzi.git makemeahanzi

    # CEDICT download and gunzip
    Ensure-EmptyDir 'cedict' | Out-Null
    $cedictGz = Join-Path $RepoRoot 'cedict_1_0_ts_utf-8_mdbg.txt.gz'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz' -OutFile $cedictGz

    # Decompress .gz to cedict/data using .NET GZipStream
    $outFile = Join-Path $RepoRoot 'cedict\data'
    $inStream = [System.IO.File]::OpenRead($cedictGz)
    try {
        $gzip = New-Object System.IO.Compression.GZipStream($inStream, [System.IO.Compression.CompressionMode]::Decompress)
        $outStream = [System.IO.File]::Create($outFile)
        try {
            $buffer = New-Object byte[] 8192
            while (($read = $gzip.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $outStream.Write($buffer, 0, $read)
            }
        } finally { $outStream.Dispose() }
    } finally {
        $inStream.Dispose()
        Remove-Item $cedictGz -Force
    }

    # TagManager assets
    New-Item -ItemType Directory -Path 'frontend/tagmanager' -Force | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/max-favilli/tagmanager/v3.0.2/tagmanager.js' -OutFile 'frontend/tagmanager/tagmanager.js'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/max-favilli/tagmanager/v3.0.2/tagmanager.css' -OutFile 'frontend/tagmanager/tagmanager.css'

    # Source Han Sans font (TC Normal). Attempt to extract with 7-Zip if available
    $fontTarget = Join-Path $RepoRoot 'SourceHanSansTC-Normal.ttf'
    if (-not (Test-Path $fontTarget)) {
        $sevenZipCmd = $null
        $sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
        if (-not $sevenZip) { $sevenZip = Get-Command 7za -ErrorAction SilentlyContinue }
        if ($sevenZip) {
            $sevenZipCmd = $sevenZip.Source
        } else {
            $fallbacks = @(
                'C:\Program Files\7-Zip\7z.exe',
                'C:\Program Files (x86)\7-Zip\7z.exe'
            )
            foreach ($p in $fallbacks) {
                if (Test-Path $p) { $sevenZipCmd = $p; break }
            }
        }

        if ($sevenZipCmd) {
            $fontArchive = Join-Path $RepoRoot 'SourceHanSansTtf.7z'
            Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/be5invis/source-han-sans-ttf/releases/download/v2.001.1/source-han-sans-ttf-2.001.1.7z' -OutFile $fontArchive
            & $sevenZipCmd e $fontArchive 'SourceHanSansTC-Normal.ttf' | Out-Null
            Remove-Item $fontArchive -Force
        } else {
            Write-Warning '7-Zip not found. Install 7-Zip and re-run to auto-fetch the font, or manually place SourceHanSansTC-Normal.ttf in the repo root.'
        }
    }

    Write-Host 'Setup complete.' -ForegroundColor Green
} finally {
    Pop-Location
}
