#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
BACKEND_DIR="${ROOT_DIR}/backend"

echo "[1/5] Creating Python virtual environment..."
python3 -m venv "${VENV_DIR}"

echo "[2/5] Installing backend dependencies..."
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${BACKEND_DIR}/requirements.txt"

if [[ ! -f "${BACKEND_DIR}/.env" ]]; then
  echo "[3/5] Creating backend/.env from example..."
  cp "${BACKEND_DIR}/.env.example" "${BACKEND_DIR}/.env"
else
  echo "[3/5] backend/.env already exists, leaving it unchanged."
fi

echo "[4/5] Backend configuration summary"
echo "  Repo:    ${ROOT_DIR}"
echo "  Backend: ${BACKEND_DIR}"
echo "  Venv:    ${VENV_DIR}"

echo "[5/5] Next steps"
echo "  1. Edit ${BACKEND_DIR}/.env if needed."
echo "  2. Ensure Ollama, ffmpeg and Java are installed on the server."
echo "  3. Start backend with:"
echo "     cd ${BACKEND_DIR} && pm2 start ecosystem.config.js"
