#!/usr/bin/env bash
# MonoceptGPT installer for macOS / Linux
set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "  MonoceptGPT Installer (macOS / Linux)"
echo "=========================================="

# 1. Python check
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.9+ from https://python.org"
  exit 1
fi
echo "[1/4] Python found: $(python3 --version)"

# 2. Virtualenv
if [ ! -d ".venv" ]; then
  echo "[2/4] Creating virtual environment (.venv)..."
  python3 -m venv .venv
else
  echo "[2/4] Virtual environment already exists"
fi

# 3. Dependencies
echo "[3/4] Installing dependencies (this may take a few minutes)..."
./.venv/bin/pip install --upgrade pip >/dev/null
./.venv/bin/pip install -r requirements.txt

# 4. Config files
echo "[4/4] Setting up config..."

if [ ! -f "teams.json" ] && [ -f "teams.json.example" ]; then
  cp teams.json.example teams.json
  echo "  - Created teams.json (from example)"
fi

if [ ! -f ".env" ]; then
  echo ""
  echo "Enter Jira credentials (press Enter to skip; you can edit .env later):"
  read -p "  JIRA_DOMAIN (e.g. your-company.atlassian.net): " D
  read -p "  JIRA_EMAIL:  " E
  read -p "  JIRA_TOKEN:  " T
  cat > .env <<EOF
JIRA_DOMAIN=$D
JIRA_EMAIL=$E
JIRA_TOKEN=$T
EOF
  chmod 600 .env
  echo "  - Created .env (chmod 600)"
else
  echo "  - .env already exists (skipped)"
fi

echo ""
echo "=========================================="
echo "  Install complete!"
echo "  Run:  ./run.sh"
echo "  Open: http://localhost:8080"
echo "=========================================="
