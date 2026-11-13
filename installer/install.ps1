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
$PythonVersion = "@PYTHON@"            # exact version, private to this install
$UvVersion = "@UV@"

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

    # ---- uv (Python package installer), a fixed version kept in bin\ ----
    $uv = "$InstallDir\bin\uv.exe"
    $have = ""
    if (Test-Path $uv) { $have = (& $uv --version) -replace '^uv\s+(\S+).*$', '$1' }
    if ($have -ne $UvVersion) {
        Write-Host "Installing uv $UvVersion (Python package installer)..."
        $env:UV_INSTALL_DIR = "$InstallDir\bin"
        $env:UV_NO_MODIFY_PATH = "1"
        $env:INSTALLER_NO_MODIFY_PATH = "1"
        Invoke-RestMethod "https://astral.sh/uv/$UvVersion/install.ps1" | Invoke-Expression
        if (-not (Test-Path $uv)) { Fail "uv could not be installed" }
    }

    # Python lives in python\ inside the install folder: never the system's,
    # not on PATH, not registered, removed by the uninstaller.
    $env:UV_PYTHON_INSTALL_DIR = "$InstallDir\python"
    $env:UV_MANAGED_PYTHON = "1"
    $env:UV_PYTHON_INSTALL_BIN = "0"
    $env:UV_PYTHON_INSTALL_REGISTRY = "0"
    $env:UV_NO_CACHE = "1"

    Write-Host "Setting up Python $PythonVersion and the required packages (this can take a few minutes)..."
    $py = "$InstallDir\venv\Scripts\python.exe"
    $current = ""
    if (Test-Path $py) { $current = (& $py -c "import platform; print(platform.python_version())") }
    if ($current -ne $PythonVersion) {
        if (Test-Path "$InstallDir\venv") { Remove-Item -Recurse -Force "$InstallDir\venv" }
        Run $uv @("venv", "--python", $PythonVersion, "$InstallDir\venv")
    }
    Run $uv @("pip", "install", "--python", $py, "--require-hashes", "-r", "$InstallDir\app\requirements.lock")

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

    $manifest = [ordered]@{ version = $Version; models = [bool]$Models; gpu = [bool]$Gpu;
                            python = "venv\Scripts\python.exe"; python_version = $PythonVersion;
                            uv = "bin\uv.exe"; uv_version = $UvVersion }
    # UTF-8 without a BOM (Set-Content -Encoding UTF8 adds one in Windows PowerShell 5.1)
    [IO.File]::WriteAllText("$InstallDir\install.json", ($manifest | ConvertTo-Json),
                            (New-Object Text.UTF8Encoding $false))

    if ($Shortcuts) {
        $ws = New-Object -ComObject WScript.Shell
        foreach ($place in @([Environment]::GetFolderPath("Programs"), [Environment]::GetFolderPath("Desktop"))) {
            $lnk = $ws.CreateShortcut("$place\Cover Cleaner.lnk")
            $lnk.TargetPath = "$InstallDir\venv\Scripts\pythonw.exe"
            $lnk.Arguments = "-m covercleaner gui"
            $lnk.WorkingDirectory = "$InstallDir\app"
            $lnk.IconLocation = "$InstallDir\app\covercleaner\assets\icon.ico,0"
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
