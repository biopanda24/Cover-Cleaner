; Cover Cleaner @VERSION@ - Windows setup (Inno Setup 6).
; Built by installer/build.py, which fills in @VERSION@ and @PAYLOAD@.

[Setup]
AppId={{6F5D3C2E-8B1A-4C4E-9E0B-3C0E4C1EA4E4}
AppName=Cover Cleaner
AppVersion=@VERSION@
AppVerName=Cover Cleaner @VERSION@
AppPublisher=Biopanda
AppPublisherURL=https://github.com/biopanda24/Cover-Cleaner
AppSupportURL=https://github.com/biopanda24/Cover-Cleaner/issues
AppUpdatesURL=https://github.com/biopanda24/Cover-Cleaner/releases
DefaultDirName={localappdata}\Programs\Cover Cleaner
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=@OUTDIR@
OutputBaseFilename=CoverCleaner-@VERSION@-windows-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=Cover Cleaner

[Tasks]
Name: "models"; Description: "Download the ML models (LaMa inpainting and disc text, about 160 MB) - recommended"
Name: "gpu"; Description: "Set up NVIDIA GPU acceleration for ML (about 1.5 GB; skipped automatically without an NVIDIA GPU) - recommended"
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "@PAYLOAD@\app\*"; DestDir: "{app}\app"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "@PAYLOAD@\install.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Cover Cleaner"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: "-m covercleaner gui"; WorkingDir: "{app}\app"; Comment: "Restore scanned CD, DVD, VHS and cassette covers"
Name: "{autodesktop}\Cover Cleaner"; Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: "-m covercleaner gui"; WorkingDir: "{app}\app"; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"" -InstallDir ""{app}"" {code:SetupFlags}"; StatusMsg: "Setting up Python, the packages and the chosen extras (this can take several minutes)..."; Flags: waituntilterminated
Filename: "{app}\venv\Scripts\pythonw.exe"; Parameters: "-m covercleaner gui"; WorkingDir: "{app}\app"; Description: "Start Cover Cleaner"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\venv.new"
Type: filesandordirs; Name: "{app}\venv.old"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\app.old"
Type: filesandordirs; Name: "{app}\bin"
Type: filesandordirs; Name: "{app}\update"
Type: files; Name: "{app}\install.json"
Type: files; Name: "{app}\*.log"

[Code]
function SetupFlags(Param: String): String;
begin
  Result := '';
  if WizardIsTaskSelected('models') then
    Result := Result + ' -Models';
  if WizardIsTaskSelected('gpu') then
    Result := Result + ' -Gpu';
end;
