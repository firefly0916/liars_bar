#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-liar_bar}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

cd "${ROOT_DIR}"

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_VLLM}" == "1" ]]; then
  conda run -n "${ENV_NAME}" python -m pip install vllm
fi

conda run -n "${ENV_NAME}" python -m pip install --no-build-isolation -e ".[llm]"

echo "Server environment is ready in conda env ${ENV_NAME}."
