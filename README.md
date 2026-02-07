# Fault Prediction Ranking Pipeline

This repository provides a reproducible Learning-to-Rank (LTR) workflow for host fault risk prediction:

1. Build query-wise ranking samples from time-stamped fault records.
2. Train a deep ranking model with PyTorch Lightning.
3. Export model artifacts and evaluation outputs for offline analysis.

The repository is prepared for open-source release with mock data templates and no production data dependency.

## Repository Layout

```text
.
├── fault_prediction/
│   ├── generate_dataset.py   # Dataset generation and feature engineering
│   ├── model.py              # LTR dataset class, sampler, and ranking model
│   └── trainer.py            # End-to-end training and evaluation pipeline
├── configs/
│   ├── train.yaml            # Model and training configuration
│   └── ltr_config.json       # Dataset generation defaults and references
├── data/
│   ├── host_fault_detail.example.json
│   ├── normal_hosts.example.json
│   └── README.md             # Data safety and usage policy
├── docs/
│   └── mock_data_schema.md   # JSON schema and examples for mock input
```

## Execution Flow

```text
Mock fault JSON
    └── fault_predicion/generate_dataset.py
            └── data/ltr_samples.pkl
                    └── scripts/train_model.py
                            ├── checkpoints/
                            ├── logs/
                            ├── model/
                            └── test_results/
```

## Environment Setup

### Option A: Local Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: Editable install with console commands

```bash
pip install -e .
```

After editable install, these commands are available:

- `fault-prediction-generate`
- `fault-prediction-train`

## Data Policy (Mock-Only in Git)

Tracked files under `data/` are mock templates only.

Before running the pipeline, create runtime copies:

```bash
cp data/host_fault_detail.example.json data/host_fault_detail.json
cp data/normal_hosts.example.json data/normal_hosts.json
```

Optional metadata file:

- `data/all_hosts_info.json` (not required; if absent, metadata fields default to `Unknown`)

Detailed schema is documented in `docs/mock_data_schema.md`.

## Usage

### 1) Generate LTR Samples

```bash
python scripts/generate_dataset.py \
  --fault-data data/host_fault_detail.json \
  --hosts-info data/all_hosts_info.json \
  --output data/ltr_samples.pkl \
  --window-days 7 \
  --step-days 3 \
  --start-date 2024-01-01 \
  --end-date 2026-01-01 \
  --min-positive-ratio 0.01 \
  --target_positive_ratio 0.1
```

If `--hosts-info` does not exist, the generator still runs.

Useful options:

- `--host-filter <keyword>`: include only hosts containing the keyword.
- `--disable_negative_sampling`: keep all negative candidates.
- `--min_negatives_per_query` and `--max_negatives_per_query`: explicit sampling bounds.

### 2) Train and Evaluate the Model

```bash
python scripts/train_model.py \
  --config configs/train.yaml \
  --data data/ltr_samples.pkl
```

### 3) Compatibility Entry Points

The following wrappers are preserved for backward compatibility:

- `python generate_optimized_ltr.py --help`
- `python deep_ltr_trainer.py --help`
- `python GenSample.py --help`

## Outputs

After a full run, the main outputs are:

- `data/ltr_samples.pkl`
- `data/ltr_samples_query_to_timestamp.json`
- `checkpoints/` (best model checkpoints)
- `logs/` (TensorBoard logs)
- `model/ltr_model.pth`
- `model/vectorizer.pkl`
- `model/scaler.pkl`
- `model/imputer.pkl`
- `model/model_metadata.json`
- `test_results/metrics.json`
- `test_results/ranking_results_<timestamp>.json`

## Reproducibility Notes

- Default random seed for negative sampling is configurable via `--sampling_seed`.
- Query-wise batching is controlled by `training.queries_per_batch` in `configs/train.yaml`.
- Temporal splitting is applied in `DeepLTRTrainer.temporal_split` to avoid random leakage.

## Docker

Build:

```bash
docker build -t fault-prediction:latest .
```

Run a quick command check:

```bash
docker run --rm fault-prediction:latest
```

## License

MIT License. See `LICENSE` for details.
