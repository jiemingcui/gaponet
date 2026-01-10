#!/bin/bash
# Sync training scripts from Isaac Lab
# Similar to humanoid_amp's sync_skrl_scripts.sh

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if ISAACLAB_PATH is set
if [ -z "$ISAACLAB_PATH" ]; then
    echo "Error: ISAACLAB_PATH environment variable is not set."
    echo "Please set it to your Isaac Lab installation path, e.g.:"
    echo "  export ISAACLAB_PATH=/path/to/isaaclab"
    exit 1
fi

# Check if Isaac Lab path exists
if [ ! -d "$ISAACLAB_PATH" ]; then
    echo "Error: Isaac Lab path does not exist: $ISAACLAB_PATH"
    exit 1
fi

# Check if training scripts exist in Isaac Lab
TRAIN_SCRIPT="$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/train.py"
PLAY_SCRIPT="$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/play.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$PLAY_SCRIPT" ]; then
    echo "Error: Play script not found: $PLAY_SCRIPT"
    exit 1
fi

# Create scripts directory
mkdir -p scripts/rsl_rl

# Copy training scripts
echo "Copying training scripts from Isaac Lab..."
cp "$TRAIN_SCRIPT" scripts/rsl_rl/
cp "$PLAY_SCRIPT" scripts/rsl_rl/

# Check if transformer training script exists
if [ -f "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/train_transformer.py" ]; then
    cp "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/train_transformer.py" scripts/rsl_rl/
    echo "  - train_transformer.py"
fi

echo "  - train.py"
echo "  - play.py"
echo ""
echo "Training scripts synced successfully!"
echo ""
echo "You can now use:"
echo "  python scripts/rsl_rl/train.py --task <your-task> ..."
echo "  python scripts/rsl_rl/play.py --task <your-task> ..."

