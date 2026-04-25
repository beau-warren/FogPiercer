# Logit Hierarchical Regression

This folder owns CDB90 ingestion, local model training, local inference checks,
and Hugging Face model storage.

The first implementation uses a multinomial/logistic regression pipeline with
hierarchical fixed effects for configurable action and group columns. Once the
exact CDB90 column names are confirmed, the config should be pinned so the
simulation emits the exact same feature labels and ordering.

## Local Setup

```bash
cd /home/fermsi/github_repos/SCSP_fogpiercer/logit_hierarchical_regression
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the root `.env.example` to `.env` at the project root and set:

- `CDB90_DATA_PATH`
- `HF_TOKEN`
- `FOGPIERCER_HF_MODEL_REPO`

## Train

Build the CDB90 model table:

```bash
python -m fogpiercer_lhr.prepare_cdb90
```

Set `CDB90_DATA_PATH` to:

```bash
logit_hierarchical_regression/data/processed/cdb90_model_table.csv
```

Then train:

```bash
python -m fogpiercer_lhr.train --config configs/train_config.yaml --push-to-hf
```

Use `--no-push-to-hf` while iterating locally.

## Local Inference Smoke Test

```bash
python -m fogpiercer_lhr.inference --model-dir models/latest --sample-row data/processed/inference_sample.json
```

