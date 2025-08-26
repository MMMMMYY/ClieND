
This repository provides the artifacts for our ASIACCS 2026 submission.  
Artifacts are shared **for availability only** (code and scripts are included).

## Contents
- `config.json` — Experiment configuration (datasets, model, FL/attack settings).
- `utils_models.py`, `data_utils.py`, `attacks.py`, `training.py`, `utils_file.py` — Model, data loading, attack, training, and I/O utilities.
- `main.py` (or your entry script) — Federated training with optional attacks; logs and results are saved under `results/`.

## Quick Start
```bash
# 1) (Optional) create env and install deps
#    Python ≥3.10, PyTorch ≥2.1 recommended
#    pip install torch torchvision pandas

# 2) Edit config.json to choose dataset/model and attack settings

# 3) Run
python main.py
