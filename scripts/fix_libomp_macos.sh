#!/bin/sh
# macOS without Homebrew: the lightgbm wheel looks for libomp in /opt/homebrew. Supply one (from any torch install, or
# pass a path) next to lib_lightgbm and add an @loader_path rpath.
set -e
SRC=${1:-$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libomp.dylib'))" 2>/dev/null)}
SP=$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])")
cp "$SRC" "$SP/lightgbm/lib/libomp.dylib"
install_name_tool -add_rpath @loader_path "$SP/lightgbm/lib/lib_lightgbm.dylib" 2>/dev/null || true
codesign -f -s - "$SP/lightgbm/lib/lib_lightgbm.dylib" "$SP/lightgbm/lib/libomp.dylib"
