#!/bin/bash

# ===============================================================================
# OpenS2S Voice Cloning - Automated Setup & Launch Script
# ===============================================================================
#
# This script automates the complete setup and launch of the OpenS2S application.
# It is designed for environments like RunPod that run as the root user by default.
#
# --- RunPod Port Configuration ---
# When creating your RunPod pod, you MUST expose the following TCP port:
#
#   - 7860: The main port for the Gradio Web UI.
#
# The other ports (21001, 21002) are for internal communication and do not
# need to be exposed to the internet.
#
# --- Usage ---
# 1. Place this script in the root of the project directory.
# 2. Make it executable: chmod +x start.sh
# 3. Run it directly:   bash start.sh
#
# ===============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONTROLLER_HOST="127.0.0.1"
CONTROLLER_PORT=21001
WORKER_HOST="127.0.0.1"
WORKER_PORT=21002
WEB_SERVER_PORT=7860
CHECKPOINTS_DIR="./checkpoints"
OPEN_S2S_DIR="$CHECKPOINTS_DIR/opens2s"
DECODER_DIR="$CHECKPOINTS_DIR/decoder"

# --- Helper Functions ---

# This function will be called when the script is terminated (e.g., with Ctrl+C)
# to ensure background processes are stopped.
cleanup() {
    echo ""
    echo "Shutting down background processes..."
    # The pkill command is more robust for finding and killing child processes.
    pkill -P $$
    echo "Shutdown complete."
}
trap cleanup EXIT

# This function waits for a network service to become available.
wait_for_service() {
    local host=$1
    local port=$2
    echo -n "Waiting for service at $host:$port to be ready..."
    # Use a loop to check with netcat until the port is open and listening.
    while ! nc -z "$host" "$port"; do
        echo -n "."
        sleep 1
    done
    echo " Service is ready."
}

# --- Main Execution ---

echo "### Step 1: Installing System Dependencies ###"
echo "(Running as root, sudo not required)"
apt-get update -y
apt-get install -y libsox-dev sox ffmpeg netcat # netcat is used for health checks

echo "
### Step 2: Installing Python Dependencies ###"
# Ensure hydra-core is in requirements, if not, add it.
grep -qxF "hydra-core" requirements.txt || echo "hydra-core" >> requirements.txt

pip install --upgrade pip
pip install -r requirements.txt
pip install -U huggingface_hub

echo "
### Step 3: Downloading Models from Hugging Face Hub ###"
echo "Creating model directories in $CHECKPOINTS_DIR..."
mkdir -p "$OPEN_S2S_DIR" "$DECODER_DIR"

# Check if models already exist to avoid re-downloading
if [ -f "$OPEN_S2S_DIR/config.json" ]; then
    echo "OpenS2S model already found. Skipping download."
else
    echo "Downloading OpenS2S model (this may take a while)..."
    huggingface-cli download CASIA-LM/OpenS2S --local-dir "$OPEN_S2S_DIR" --local-dir-use-symlinks False
fi

if [ -f "$DECODER_DIR/config.yaml" ]; then
    echo "GLM-4 Voice Decoder model already found. Skipping download."
else
    echo "Downloading GLM-4 Voice Decoder model..."
    huggingface-cli download THUDM/glm-4-voice-decoder --local-dir "$DECODER_DIR" --local-dir-use-symlinks False
fi

echo "
### Step 4: Launching Application Services ###"

echo "--> Starting controller in the background..."
python3 controller.py --host $CONTROLLER_HOST --port $CONTROLLER_PORT &
wait_for_service $CONTROLLER_HOST $CONTROLLER_PORT

echo "--> Starting model worker in the background..."
python3 model_worker.py --host $WORKER_HOST --port $WORKER_PORT --controller-address http://$CONTROLLER_HOST:$CONTROLLER_PORT --model-path "$OPEN_S2S_DIR" --flow-path "$DECODER_DIR" &
wait_for_service $WORKER_HOST $WORKER_PORT

echo "--> Starting Gradio web UI in the foreground..."
echo ""

echo "==============================================================================="
echo "  Application is now running!"
echo ""
echo "  Access the web interface at: http://127.0.0.1:$WEB_SERVER_PORT"
echo "  (If on RunPod, use the public URL provided by the service)"
echo "==============================================================================="
echo ""
echo "Press Ctrl+C to stop all services."
python3 web_demo.py --port $WEB_SERVER_PORT --host 0.0.0.0 --share