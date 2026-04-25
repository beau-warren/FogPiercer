# SCSP Fogpiercer

Local hackathon demo for a national defense AI competition. The project has two
parts:

- `logit_hierarchical_regression/`: training, local inference tests, feature
  schema, data staging, and final trained model artifacts.
- `simulation/`: local 2D battlefield simulation and UI that pulls model
  outputs from the trained model endpoint/artifact.

## Safety and Secrets

This repo is intended to be safe to publish later. Do not commit `.env`, API
keys, raw restricted datasets, generated model binaries, or Hugging Face tokens.
Use `.env.example` as the source of truth for variable names.

Required local environment variables:

- `MERCURY_II_API_KEY`: API key for Inception Mercury II.
- `MERCURY_II_BASE_URL`: Mercury II API base URL, if different from the
  default configured later.
- `HF_TOKEN`: Hugging Face token with write access for training pushes and read
  access for demo pulls.
- `FOGPIERCER_HF_MODEL_REPO`: Hugging Face model repository, for example
  `your-hf-username/SCSP_fogpiercer-logit-hierarchical-regression`.
- `GITHUB_TOKEN`: optional, only for later GitHub automation.
- `CDB90_DATA_PATH`: local path to the CDB90 CSV.

## Step Checkpoints

1. Train and validate the Logit Hierarchical Regression model on local CDB90
   data, then push the trained artifact to Hugging Face.
2. Build the simple 2D visual environment with realistic unit symbology and
   drag-and-drop unit movement.
3. Generate synthetic sensor data from live simulation state using the same
   feature schema as the CDB90 training data.
4. Ask Mercury II to summarize the top 3 model-ranked decisions in plain
   operator-friendly language.
5. End/reset the demo when one side is removed, preserving the final decision
   and requiring a user click to reiterate.

