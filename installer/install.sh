#!/usr/bin/env bash
# Cover Cleaner @VERSION@ installer for @OS_NAME@.
# Self-extracting: the program files follow the __PAYLOAD_BELOW__ line.
#
#   bash CoverCleaner-@VERSION@-@OS@.run [options]
#     --dir DIR      install folder (default: see below)
#     --yes          use the recommended answers without asking
#     --no-models    do not download the ML models
#     --no-gpu       do not set up NVIDIA GPU support
#     --uninstall    remove an installed copy
set -euo pipefail

APP_NAME="Cover Cleaner"
VERSION="@VERSION@"
OS_KIND="@OS@"                         # linux | macos
PYTHON_VERSION="3.11"

if [ "$OS_KIND" = "macos" ]; then
  DIR="$HOME/Library/Application Support/Cover Cleaner"
else
  DIR="${XDG_DATA_HOME:-$HOME/.local/share}/cover-cleaner"
fi
ASK=1; MODELS=""; GPU=""; UNINSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --yes|-y) ASK=0; shift ;;
    --no-models) MODELS=0; shift ;;
    --no-gpu) GPU=0; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*"; }
ask_yes() {   # ask_yes "question" -> 0 for yes (default yes)
  if [ "$ASK" = 0 ] || { [ ! -t 0 ] && [ ! -r /dev/tty ]; }; then return 0; fi
  local reply
  printf '%s [Y/n] ' "$1"
  read -r reply </dev/tty || reply=""
  case "$reply" in [nN]*) return 1 ;; *) return 0 ;; esac
}

launcher_paths() {
  if [ "$OS_KIND" = "macos" ]; then
    APP_BUNDLE="$HOME/Applications/Cover Cleaner.app"
  else
    BIN_LINK="$HOME/.local/bin/cover-cleaner"
    DESKTOP="${XDG_DATA_HOME:-$HOME/.local/share}/applications/cover-cleaner.desktop"
  fi
}

if [ "$UNINSTALL" = 1 ]; then
  launcher_paths
  say "Removing $APP_NAME from $DIR"
  rm -rf "$DIR/app" "$DIR/app.old" "$DIR/venv" "$DIR/bin" "$DIR/update" "$DIR/install.json" \
         "$DIR/cover-cleaner" "$DIR/uninstall.sh" "$DIR"/*.log
  rmdir "$DIR" 2>/dev/null || true
  if [ "$OS_KIND" = "macos" ]; then rm -rf "$APP_BUNDLE"; else rm -f "$BIN_LINK" "$DESKTOP"; fi
  say "Done. Your options (settings.json) are kept in your user settings folder."
  exit 0
fi

say "== $APP_NAME $VERSION =="
say "Install folder: $DIR"

# ---- choices --------------------------------------------------------------
if [ -z "$MODELS" ]; then
  if ask_yes "Download the ML models (LaMa inpainting and disc text, about 160 MB) - recommended?"; then MODELS=1; else MODELS=0; fi
fi
ARCH="$(uname -m)"
if [ -z "$GPU" ]; then
  GPU=0
  if [ "$OS_KIND" = "linux" ] && [ "$ARCH" = "x86_64" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      if ask_yes "Set up NVIDIA GPU acceleration for ML (about 1.5 GB download) - recommended?"; then GPU=1; fi
    else
      say "No NVIDIA GPU found: ML will run on the CPU."
    fi
  fi
fi

# ---- unpack ---------------------------------------------------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
LINE=$(awk '/^__PAYLOAD_BELOW__$/ { print NR + 1; exit 0 }' "$0")
tail -n +"$LINE" "$0" | tar -xzf - -C "$TMP"

mkdir -p "$DIR"
if [ -d "$DIR/app" ]; then                     # reinstall / upgrade: keep the models
  rm -rf "$DIR/app.old"
  mv "$DIR/app" "$DIR/app.old"
fi
cp -R "$TMP/app" "$DIR/app"
if [ -d "$DIR/app.old/models" ] && [ ! -d "$DIR/app/models" ]; then
  mv "$DIR/app.old/models" "$DIR/app/models"
fi
rm -rf "$DIR/app.old"

# ---- Python and packages (via uv) -----------------------------------------
if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  UV="$DIR/bin/uv"
  if [ ! -x "$UV" ]; then
    say "Installing uv (Python package installer)..."
    if command -v curl >/dev/null 2>&1; then FETCH="curl -LsSf"; else FETCH="wget -qO-"; fi
    $FETCH https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$DIR/bin" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh
  fi
fi
say "Setting up Python $PYTHON_VERSION and the required packages (this can take a few minutes)..."
"$UV" venv --python "$PYTHON_VERSION" --allow-existing "$DIR/venv"
PY="$DIR/venv/bin/python"
"$UV" pip install --python "$PY" -r "$DIR/app/requirements.txt"

export PYTHONPATH="$DIR/app"
export COVERCLEANER_UV="$UV"
if [ "$MODELS" = 1 ]; then "$PY" -m covercleaner.setup_models --models || say "Some models could not be downloaded; run setup again later."; fi
if [ "$GPU" = 1 ]; then "$PY" -m covercleaner.setup_models --gpu || say "GPU setup failed; ML will run on the CPU."; fi

UV_REL="$UV"
case "$UV" in "$DIR"/*) UV_REL="${UV#"$DIR"/}" ;; esac
"$PY" - "$DIR/install.json" "$VERSION" "$MODELS" "$GPU" "$UV_REL" <<'PYEOF'
import json, sys
path, version, models, gpu, uv = sys.argv[1:6]
json.dump({"version": version, "models": models == "1", "gpu": gpu == "1",
           "python": "venv/bin/python", "uv": uv}, open(path, "w"), indent=2)
PYEOF

# ---- launchers ------------------------------------------------------------
cat > "$DIR/cover-cleaner" <<EOF
#!/usr/bin/env bash
export PYTHONPATH="$DIR/app"
exec "$PY" -m covercleaner gui "\$@"
EOF
chmod +x "$DIR/cover-cleaner"
cp "$0" "$DIR/uninstall.sh" 2>/dev/null || true
chmod +x "$DIR/uninstall.sh" 2>/dev/null || true

launcher_paths
if [ "$OS_KIND" = "macos" ]; then
  mkdir -p "$APP_BUNDLE/Contents/MacOS"
  cat > "$APP_BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Cover Cleaner</string>
  <key>CFBundleDisplayName</key><string>Cover Cleaner</string>
  <key>CFBundleIdentifier</key><string>io.github.biopanda24.covercleaner</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleExecutable</key><string>cover-cleaner</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOF
  cat > "$APP_BUNDLE/Contents/MacOS/cover-cleaner" <<EOF
#!/usr/bin/env bash
exec "$DIR/cover-cleaner" "\$@"
EOF
  chmod +x "$APP_BUNDLE/Contents/MacOS/cover-cleaner"
  say "Installed. Open 'Cover Cleaner' from ~/Applications (or run: \"$DIR/cover-cleaner\")."
else
  mkdir -p "$(dirname "$BIN_LINK")" "$(dirname "$DESKTOP")"
  ln -sf "$DIR/cover-cleaner" "$BIN_LINK"
  cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=Cover Cleaner
Comment=Restore scanned CD, DVD, VHS and cassette covers
Exec="$DIR/cover-cleaner" %F
Terminal=false
Categories=Graphics;Photography;
MimeType=image/png;image/jpeg;image/tiff;
EOF
  if ! ldconfig -p 2>/dev/null | grep -q libxcb-cursor; then
    say "Note: the Qt interface needs the 'libxcb-cursor0' system library"
    say "      (e.g. sudo apt install libxcb-cursor0) if the window does not open."
  fi
  say "Installed. Start 'Cover Cleaner' from your applications menu, or run: cover-cleaner"
fi
say "To uninstall: bash \"$DIR/uninstall.sh\" --uninstall"
exit 0
__PAYLOAD_BELOW__
