
This repository provides the artifacts for our ASIACCS 2026 submission.  

## Contents
- `config.json` — Experiment configuration (datasets, model, FL/attack settings).
- `utils_models.py`, `data_utils.py`, `attacks.py`, `training.py`, `utils_file.py` — Model, data loading, attack, training, and I/O utilities.
- `main.py` (or your entry script) — Federated training with optional attacks; logs and results are saved under `results/`.
- `aggregatio.py` - Include the server defense, such as FLTrust, FreqFed, CrowdGuard, FLSheld, Flame, and FedDefender.

## Quick Start

# 1) Create Environment and Install Dependencies

Recommended: Python ≥3.10, PyTorch ≥2.1.

```bash
pip install torch torchvision pandas
```

# 2) Edit Configuration

Edit `config.json` to choose the dataset, model, and attack settings.

# 3) Run

```bash
python main.py
```

All the training details will be saved.

# 4) Result Saving

The experimental results are automatically saved in the `results/{file_suffix}/` directory, where `{file_suffix}` acts as a unique identifier for the specific run configuration.

The following CSV files are generated:

* **Client Metrics**:
  * `client_{client_id}_scores.csv`: Contains the detailed scoring metrics for each individual client.
  * `client_{client_id}_weights.csv`: Stores the local model weights for each client.
  * `client_{client_id}_accuracies.csv`: Records the local accuracy history for each client over the training rounds.

* **Server Metrics**:
  * `server_scores.csv`: Logs the aggregated scores tracked by the central server.
  * `server_weights.csv`: Saves the aggregated global model weights.
  * `server_accuracies.csv`: Records the global model's accuracy history (aggregated performance) throughout the training process.

# 5) ClieND

Run `cliend.py` to execute the whole process of client detection and mitigation.

```bash
python cliend.py
```
