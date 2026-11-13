# Cover Cleaner @VERSION@ - Windows setup.
# Run by the setup program after it has copied the files (or by
# "Install Cover Cleaner.cmd" from the zip). Sets up Python and the
# packages with uv, optionally downloads the ML models and NVIDIA GPU
# support, and writes install.json.
param(
    [string]$InstallDir = "$env:LOCALAPPDATA\Programs\Cover Cleaner",
    [string]$AppSource = "",          # zip install: folder holding app\
    [switch]$Models,
    [switch]$Gpu,
    [switch]$Shortcuts                # zip install: create Start menu / desktop shortcuts
)
$ErrorActionPreference = "Stop"
$Version = "@VERSION@"
$PythonVersion = "3.11"

function Fail($msg) {
    Write-Host ""
    Write-Host "SETUP FAILED: $msg" -ForegroundColor Red
    Write-Host "See $InstallDir\install.log. Press Enter to close."
    try { Stop-Transcript | Out-Null } catch {}
    [void](Read-Host)
    exit 1
}
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { Fail "$exe $($arguments -join ' ') (exit $LASTEXITCODE)" }
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
try { Start-Transcript -Path "$InstallDir\install.log" -Append | Out-Null } catch {}
Write-Host "== Cover Cleaner $Version =="
Write-Host "Install folder: $InstallDir"

try {
    if ($AppSource) {                                   # zip install: copy the program
        if (Test-Path "$InstallDir\app") {
            if (Test-Path "$InstallDir\app.old") { Remove-Item -Recurse -Force "$InstallDir\app.old" }
            Rename-Item "$InstallDir\app" "app.old"
        }
        Copy-Item -Recurse "$AppSource\app" "$InstallDir\app"
        if ((Test-Path "$InstallDir\app.old\models") -and -not (Test-Path "$InstallDir\app\models")) {
            Move-Item "$InstallDir\app.old\models" "$InstallDir\app\models"
        }
        if (Test-Path "$InstallDir\app.old") { Remove-Item -Recurse -Force "$InstallDir\app.old" }
        Copy-Item -Force "$PSCommandPath" "$InstallDir\install.ps1" -ErrorAction SilentlyContinue
    }

    # ---- uv (Python package installer) --------------------------------
    $uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
    if (-not $uv) {
        $uv = "$InstallDir\bin\uv.exe"
        if (-not (Test-Path $uv)) {
            Write-Host "Installing uv (Python package installer)..."
            $env:UV_INSTALL_DIR = "$InstallDir\bin"
            $env:UV_NO_MODIFY_PATH = "1"
            $env:INSTALLER_NO_MODIFY_PATH = "1"
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
            if (-not (Test-Path $uv)) { Fail "uv could not be installed" }
        }
    }

    Write-Host "Setting up Python $PythonVersion and the required packages (this can take a few minutes)..."
    Run $uv @("venv", "--python", $PythonVersion, "--allow-existing", "$InstallDir\venv")
    $py = "$InstallDir\venv\Scripts\python.exe"
    Run $uv @("pip", "install", "--python", $py, "-r", "$InstallDir\app\requirements.txt")

    $env:PYTHONPATH = "$InstallDir\app"
    $env:COVERCLEANER_UV = $uv
    if ($Models) {
        & $py -m covercleaner.setup_models --models
        if ($LASTEXITCODE -ne 0) { Write-Host "Some models could not be downloaded; run setup again later." }
    }
    if ($Gpu) {
        & $py -m covercleaner.setup_models --gpu
        if ($LASTEXITCODE -ne 0) { Write-Host "GPU setup failed; ML will run on the CPU." }
    }

    $uvRel = $uv
    if ($uv.StartsWith("$InstallDir\")) { $uvRel = $uv.Substring($InstallDir.Length + 1) }
    $manifest = [ordered]@{ version = $Version; models = [bool]$Models; gpu = [bool]$Gpu;
                            python = "venv\Scripts\python.exe"; uv = $uvRel }
    $manifest | ConvertTo-Json | Set-Content -Encoding UTF8 "$InstallDir\install.json"

    if ($Shortcuts) {
        $ws = New-Object -ComObject WScript.Shell
        foreach ($place in @([Environment]::GetFolderPath("Programs"), [Environment]::GetFolderPath("Desktop"))) {
            $lnk = $ws.CreateShortcut("$place\Cover Cleaner.lnk")
            $lnk.TargetPath = "$InstallDir\venv\Scripts\pythonw.exe"
            $lnk.Arguments = "-m covercleaner gui"
            $lnk.WorkingDirectory = "$InstallDir\app"
            $lnk.Description = "Restore scanned CD, DVD, VHS and cassette covers"
            $lnk.Save()
        }
    }
} catch {
    Fail $_.Exception.Message
}
Write-Host ""
Write-Host "Cover Cleaner $Version is installed." -ForegroundColor Green
try { Stop-Transcript | Out-Null } catch {}
exit 0
