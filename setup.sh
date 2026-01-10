#!/bin/bash
# GapONet Setup Script
# This script sets up the conda environment and installs all dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default environment name
ENV_NAME="${1:-gapo}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GapONet Setup Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}[ERROR] Conda could not be found. Please install conda and try again.${NC}"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if the environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}[INFO] Conda environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}[INFO] Removing existing environment '${ENV_NAME}'...${NC}"
        conda env remove -n "${ENV_NAME}" -y
    else
        echo -e "${GREEN}[INFO] Using existing environment '${ENV_NAME}'.${NC}"
    fi
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${GREEN}[INFO] Creating conda environment '${ENV_NAME}' with Python 3.10...${NC}"
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate the environment
echo -e "${GREEN}[INFO] Activating conda environment '${ENV_NAME}'...${NC}"
conda activate "${ENV_NAME}"

# Install conda packages
echo -e "${GREEN}[INFO] Installing conda packages...${NC}"
conda install -c conda-forge -y \
    numpy \
    scipy \
    matplotlib \
    ipython \
    jupyter \
    importlib_metadata

# Install PyTorch (with CUDA support if available)
echo -e "${GREEN}[INFO] Installing PyTorch...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}[INFO] CUDA detected. Installing PyTorch with CUDA support...${NC}"
    conda install -c pytorch -c nvidia -y \
        pytorch \
        torchvision \
        torchaudio \
        pytorch-cuda=11.8 || \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}[INFO] No CUDA detected. Installing CPU-only PyTorch...${NC}"
    conda install -c pytorch -y \
        pytorch \
        torchvision \
        torchaudio \
        cpuonly || \
    pip install torch torchvision torchaudio
fi

# Upgrade pip
echo -e "${GREEN}[INFO] Upgrading pip...${NC}"
pip install --upgrade pip

# Install Python dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}[INFO] Installing Python dependencies from requirements.txt...${NC}"
    pip install -r requirements.txt
else
    echo -e "${YELLOW}[WARNING] requirements.txt not found. Installing dependencies manually...${NC}"
    pip install \
        numpy>=1.21.0 \
        gymnasium>=0.28.0 \
        pinocchio>=2.6.0 \
        pytorch-kinematics>=0.0.1 \
        psutil>=5.9.0 \
        toml>=0.10.2
fi

# Explicitly install pytorch-kinematics to ensure it's available
echo -e "${GREEN}[INFO] Installing pytorch-kinematics...${NC}"
pip install pytorch-kinematics>=0.0.1 || echo -e "${YELLOW}[WARNING] Failed to install pytorch-kinematics. You may need to install it manually.${NC}"

# Install optional dependencies (uncomment if needed)
echo -e "${GREEN}[INFO] Installing optional dependencies...${NC}"
pip install \
    wandb>=0.15.0 \
    tensorboard>=2.13.0 \
    matplotlib>=3.5.0

# Install GapONet package in editable mode
echo -e "${GREEN}[INFO] Installing GapONet package in editable mode...${NC}"
pip install -e .

# Sync training scripts (optional)
if [ ! -z "$ISAACLAB_PATH" ] && [ -d "$ISAACLAB_PATH" ]; then
    echo -e "${GREEN}[INFO] Syncing training scripts from Isaac Lab...${NC}"
    ./sync_rsl_scripts.sh || echo -e "${YELLOW}[WARNING] Failed to sync scripts. You can run './sync_rsl_scripts.sh' manually later.${NC}"
else
    echo -e "${YELLOW}[INFO] ISAACLAB_PATH not set. Skipping script sync.${NC}"
    echo -e "${YELLOW}[INFO] To sync training scripts later, set ISAACLAB_PATH and run: ./sync_rsl_scripts.sh${NC}"
fi

# Setup conda environment activation script
echo -e "${GREEN}[INFO] Setting up conda environment activation scripts...${NC}"
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${CONDA_PREFIX}/etc/conda/deactivate.d"

# Create activation script
cat > "${CONDA_PREFIX}/etc/conda/activate.d/gaponet_setenv.sh" << EOF
#!/bin/bash
# GapONet environment activation script

# Set project path
export GAPONET_PATH="${SCRIPT_DIR}"

# Add to PYTHONPATH if needed
# export PYTHONPATH="\${PYTHONPATH}:\${GAPONET_PATH}"

echo "[GapONet] Environment activated. Project path: \${GAPONET_PATH}"
EOF

# Create deactivation script
cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/gaponet_unsetenv.sh" << EOF
#!/bin/bash
# GapONet environment deactivation script

unset GAPONET_PATH

echo "[GapONet] Environment deactivated."
EOF

chmod +x "${CONDA_PREFIX}/etc/conda/activate.d/gaponet_setenv.sh"
chmod +x "${CONDA_PREFIX}/etc/conda/deactivate.d/gaponet_unsetenv.sh"

# Reactivate to load new environment variables
conda deactivate
conda activate "${ENV_NAME}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "To activate the environment, run:"
echo -e "  ${GREEN}conda activate ${ENV_NAME}${NC}"
echo -e ""
echo -e "To verify installation, run:"
echo -e "  ${GREEN}python -c 'import torch; import numpy; import gymnasium; print(\"All packages installed successfully!\")'${NC}"
echo -e ""

