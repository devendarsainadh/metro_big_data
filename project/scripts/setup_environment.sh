#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  conda env create -f environment.yml || conda env update -f environment.yml
  echo "Conda environment ready: metro-data-pipeline"
else
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install pandas numpy scikit-learn pyyaml pyspark jupyterlab ucimlrepo
  echo "Virtualenv ready in .venv"
fi
