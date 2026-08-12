#!/usr/bin/env bash
# ============================================================================
# Setup script for AgentDAM defender environment.
#
# This script installs all dependencies needed to run the AgentDAM-based
# defender with real websites (Playwright browser + Docker containers).
#
# Usage:
#     bash scripts/setup_agentdam_env.sh [--no-docker] [--no-playwright]
#
# What this does:
#   1. Installs Python packages (playwright, beartype, gymnasium, etc.)
#   2. Installs Playwright Chromium browser
#   3. Sets up environment variables for website URLs
#   4. (Optional) Pulls and starts Docker containers for real websites
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AGENTDAM_ROOT="/home/fangzibang/ai-agent-privacy"
VWA_ROOT="$AGENTDAM_ROOT/visualwebarena"
CONDA_ENV="AgentPrivacy"

NO_DOCKER=false
NO_PLAYWRIGHT=false

for arg in "$@"; do
    case "$arg" in
        --no-docker) NO_DOCKER=true ;;
        --no-playwright) NO_PLAYWRIGHT=true ;;
    esac
done

echo "============================================"
echo "AgentDAM Defender Environment Setup"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# 1. Install Python dependencies
# ------------------------------------------------------------------
echo "[1/4] Installing Python packages into conda env '$CONDA_ENV'..."

conda run -n "$CONDA_ENV" pip install --quiet \
    playwright==1.37.0 \
    beartype==0.12.0 \
    gymnasium==0.29.1 \
    aiolimiter==1.2.1 \
    evaluate==0.4.0 \
    scikit-image==0.22.0 \
    matplotlib==3.8.0 \
    nltk==3.8.1 \
    tiktoken==0.8.0 \
    2>&1 | tail -3

echo "   Python packages installed."

# ------------------------------------------------------------------
# 2. Install Playwright browsers
# ------------------------------------------------------------------
if [ "$NO_PLAYWRIGHT" = false ]; then
    echo ""
    echo "[2/4] Installing Playwright Chromium browser..."
    conda run -n "$CONDA_ENV" python -m playwright install chromium 2>&1 | tail -3
    echo "   Chromium browser installed."
else
    echo ""
    echo "[2/4] Skipping Playwright browser install (--no-playwright)."
fi

# ------------------------------------------------------------------
# 3. Set up AgentDAM as importable package
# ------------------------------------------------------------------
echo ""
echo "[3/4] Configuring AgentDAM path..."

# Add to conda env's site-packages via .pth file
SITE_PACKAGES=$(conda run -n "$CONDA_ENV" python -c "import site; print(site.getsitepackages()[0])")
echo "$VWA_ROOT" > "$SITE_PACKAGES/agentdam_vwa.pth"
echo "$AGENTDAM_ROOT" > "$SITE_PACKAGES/agentdam_root.pth"
echo "   Added .pth files to $SITE_PACKAGES"
echo "   VisualWebArena path: $VWA_ROOT"
echo "   AgentDAM path: $AGENTDAM_ROOT"

# ------------------------------------------------------------------
# 4. Docker containers for real websites
# ------------------------------------------------------------------
echo ""
if [ "$NO_DOCKER" = true ]; then
    echo "[4/4] Skipping Docker setup (--no-docker)."
    echo ""
    echo "To run with real websites later:"
    echo "  1. Start Docker daemon:  sudo service docker start"
    echo "  2. Download Docker images (see $VWA_ROOT/environment_docker/README.md)"
    echo "  3. Start containers with the reset scripts."
else
    echo "[4/4] Docker website containers..."

    # Check Docker
    if ! command -v docker &>/dev/null; then
        echo "   ERROR: Docker not found. Install Docker first."
        echo "   Then re-run with: bash scripts/setup_agentdam_env.sh"
        exit 1
    fi

    if ! docker ps &>/dev/null; then
        echo "   Docker daemon not running. Attempting to start..."
        sudo service docker start 2>/dev/null || {
            echo "   ERROR: Cannot start Docker daemon. Start it manually:"
            echo "          sudo service docker start"
            echo "   Then re-run: bash scripts/setup_agentdam_env.sh"
            exit 1
        }
        sleep 2
    fi

    # Check if website containers exist
    SHOPPING_RUNNING=$(docker ps --filter "name=shopping" --format "{{.Names}}" 2>/dev/null || true)
    REDDIT_RUNNING=$(docker ps --filter "name=reddit" --format "{{.Names}}" 2>/dev/null || true)
    GITLAB_RUNNING=$(docker ps --filter "name=gitlab" --format "{{.Names}}" 2>/dev/null || true)

    if [ -z "$SHOPPING_RUNNING" ]; then
        echo "   WARNING: Shopping container not running."
        echo "   To start: docker run -d --name shopping -p 7770:80 shopping_final_0712"
    else
        echo "   Shopping container: $SHOPPING_RUNNING"
    fi

    if [ -z "$REDDIT_RUNNING" ]; then
        echo "   WARNING: Reddit container not running."
        echo "   To start: docker run -d --name reddit -p 9999:80 postmill-populated-exposed-withimg"
    else
        echo "   Reddit container: $REDDIT_RUNNING"
    fi

    if [ -z "$GITLAB_RUNNING" ]; then
        echo "   WARNING: GitLab container not running."
        echo "   To start: docker run -d --name gitlab -p 8023:80 gitlab-populated-final-port8023"
    else
        echo "   GitLab container: $GITLAB_RUNNING"
    fi
fi

# ------------------------------------------------------------------
# 5. Environment variables
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "Environment Variables (add to your shell profile):"
echo "============================================"
echo ""
cat << 'EOF'
export DATASET=visualwebarena
export SHOPPING="http://localhost:7770"
export REDDIT="http://localhost:9999"
export GITLAB="http://localhost:8023"
export WIKIPEDIA="http://localhost:8888"
export HOMEPAGE="http://localhost:4399"
export CLASSIFIEDS="http://localhost:9980"
EOF

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Quick test (mock mode, no Docker needed):"
echo "  conda run -n $CONDA_ENV python tests/test_agentdam_defender.py \\"
echo "      --model_path /path/to/model --port 5000 --gpu_id 0 --mock"
echo ""
echo "Full test (requires Docker + real websites):"
echo "  conda run -n $CONDA_ENV python tests/test_agentdam_defender.py \\"
echo "      --model_path /path/to/model --port 5000 --gpu_id 0 \\"
echo "      --site reddit --max_cases 4"
echo "============================================"
