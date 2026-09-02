```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Jetson Orin Nano 8GB
# NVIDIA NanoLLM / Agent Studio Launcher
#
# Directory:
# jetson_prep/
# ├── start_agent_studio.sh
# ├── AutoPrompt.txt
# ├── Chat Node.txt
# ├── plugins/
# │   ├── __init__.py
# │   └── custom_alert_tool.py
# └── videos/
#     └── test_video.mp4
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo
echo "============================================================"
echo " NVIDIA Jetson Orin Nano - NanoLLM Agent Studio"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. Hugging Face token
# ------------------------------------------------------------

if [[ -z "${HUGGINGFACE_TOKEN:-}" ]]; then
    echo "ERROR: HUGGINGFACE_TOKEN is not set."
    echo
    echo "Run:"
    echo '  export HUGGINGFACE_TOKEN="hf_your_new_token_here"'
    echo
    exit 1
fi

echo "[1/6] Hugging Face token found."

# ------------------------------------------------------------
# 2. Check available memory/swap
# ------------------------------------------------------------

echo
echo "[2/6] Checking Jetson memory..."

TOTAL_RAM_MB=$(free -m | awk '/^Mem:/ {print $2}')
SWAP_TOTAL_MB=$(free -m | awk '/^Swap:/ {print $2}')

echo "RAM : ${TOTAL_RAM_MB} MB"
echo "Swap: ${SWAP_TOTAL_MB} MB"

# Do NOT automatically create a huge swapfile on the SD card.
# Existing JetPack swap/zram is left untouched.

if [[ "$SWAP_TOTAL_MB" -lt 4096 ]]; then
    echo
    echo "WARNING: Less than 4GB swap is available."
    echo "The Jetson may run out of memory with larger AI models."
    echo "If required, configure additional swap separately."
else
    echo "Existing swap is sufficient for initial setup."
fi

# ------------------------------------------------------------
# 3. Setup jetson-containers
# ------------------------------------------------------------

CONTAINERS_DIR="$SCRIPT_DIR/jetson-containers"

echo
echo "[3/6] Checking jetson-containers..."

if [[ ! -d "$CONTAINERS_DIR" ]]; then
    echo "Cloning jetson-containers..."

    git clone --depth=1 \
        https://github.com/dusty-nv/jetson-containers \
        "$CONTAINERS_DIR"
else
    echo "jetson-containers already exists."
fi

# The installer normally places the commands in PATH.
# Run it only if the commands are missing.

if ! command -v jetson-containers >/dev/null 2>&1 || \
   ! command -v autotag >/dev/null 2>&1; then

    echo "Installing jetson-containers tools..."

    bash "$CONTAINERS_DIR/install.sh"
fi

# Refresh PATH in case installer added /usr/local/bin.
export PATH="/usr/local/bin:$PATH"

if ! command -v jetson-containers >/dev/null 2>&1; then
    echo "ERROR: jetson-containers command not found."
    exit 1
fi

if ! command -v autotag >/dev/null 2>&1; then
    echo "ERROR: autotag command not found."
    exit 1
fi

echo "jetson-containers: OK"
echo "autotag:            OK"

# ------------------------------------------------------------
# 4. Get NanoLLM source
# ------------------------------------------------------------

NANOLLM_DIR="$SCRIPT_DIR/NanoLLM"

echo
echo "[4/6] Checking NanoLLM..."

if [[ ! -d "$NANOLLM_DIR" ]]; then
    echo "Cloning NanoLLM..."

    git clone --depth=1 \
        https://github.com/dusty-nv/NanoLLM \
        "$NANOLLM_DIR"
else
    echo "NanoLLM already exists."
fi

# ------------------------------------------------------------
# 5. Install custom SafetyAlertPlugin
# ------------------------------------------------------------

echo
echo "[5/6] Installing SafetyAlertPlugin..."

PLUGIN_SRC="$SCRIPT_DIR/plugins/custom_alert_tool.py"
PLUGIN_DST="$NANOLLM_DIR/nano_llm/plugins/custom_alert_tool.py"
INIT_FILE="$NANOLLM_DIR/nano_llm/plugins/__init__.py"

if [[ ! -f "$PLUGIN_SRC" ]]; then
    echo "ERROR: Custom plugin not found:"
    echo "  $PLUGIN_SRC"
    exit 1
fi

if [[ ! -f "$INIT_FILE" ]]; then
    echo "ERROR: NanoLLM plugin __init__.py not found:"
    echo "  $INIT_FILE"
    exit 1
fi

mkdir -p "$(dirname "$PLUGIN_DST")"

# Copy/update custom plugin.
cp "$PLUGIN_SRC" "$PLUGIN_DST"

# Register plugin only once.
if ! grep -qF \
    "from .custom_alert_tool import SafetyAlertPlugin" \
    "$INIT_FILE"; then

    echo \
        "from .custom_alert_tool import SafetyAlertPlugin" \
        >> "$INIT_FILE"

    echo "SafetyAlertPlugin registered."
else
    echo "SafetyAlertPlugin already registered."
fi

# ------------------------------------------------------------
# 6. Prepare storage and launch
# ------------------------------------------------------------

echo
echo "[6/6] Preparing persistent storage..."

VIDEOS_DIR="$SCRIPT_DIR/videos"
MODELS_DIR="$SCRIPT_DIR/data/models"
PRESETS_DIR="$SCRIPT_DIR/data/presets"

mkdir -p "$VIDEOS_DIR"
mkdir -p "$MODELS_DIR"
mkdir -p "$PRESETS_DIR"

if [[ ! -f "$VIDEOS_DIR/test_video.mp4" ]]; then
    echo
    echo "WARNING:"
    echo "No test_video.mp4 found."
    echo "Place your test video at:"
    echo "  $VIDEOS_DIR/test_video.mp4"
fi

# ------------------------------------------------------------
# Find compatible NanoLLM container
# ------------------------------------------------------------

echo
echo "Finding compatible NanoLLM container..."

NANO_IMAGE="$(autotag nano_llm)"

if [[ -z "$NANO_IMAGE" ]]; then
    echo "ERROR: Could not determine a NanoLLM container image."
    exit 1
fi

echo
echo "Container:"
echo "  $NANO_IMAGE"

# ------------------------------------------------------------
# Launch Agent Studio
# ------------------------------------------------------------

echo
echo "============================================================"
echo " Starting NVIDIA Agent Studio"
echo "============================================================"
echo
echo "Open from another computer:"
echo
echo "  http://<JETSON-IP>:8050"
echo
echo "Example:"
echo "  http://192.168.1.100:8050"
echo
echo "Press Ctrl+C to stop Agent Studio."
echo

exec jetson-containers run \
    --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
    --volume "$NANOLLM_DIR:/opt/NanoLLM" \
    --volume "$VIDEOS_DIR:/data/videos" \
    --volume "$MODELS_DIR:/root/.cache/huggingface" \
    --volume "$PRESETS_DIR:/data/nano_llm/presets" \
    "$NANO_IMAGE" \
    python3 -m nano_llm.studio \
        --host 0.0.0.0 \
        --web-port 8050
```
