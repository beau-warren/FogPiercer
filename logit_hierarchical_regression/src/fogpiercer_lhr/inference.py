from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from fogpiercer_lhr.config import env_value, load_project_env
from fogpiercer_lhr.hf_io import download_model_dir


def _json_safe(value: object) -> object:
    if pd.isna(value):
        return None
    return value


def load_model(model_dir: Path | None) -> tuple[object, dict[str, object]]:
    load_project_env()
    resolved_dir = model_dir
    if resolved_dir is None:
        repo_id = env_value("FOGPIERCER_HF_MODEL_REPO")
        token = env_value("HF_TOKEN", required=False)
        resolved_dir = download_model_dir(repo_id=repo_id, token=token)

    pipeline = joblib.load(resolved_dir / "model.joblib")
    schema = json.loads((resolved_dir / "feature_schema.json").read_text(encoding="utf-8"))
    return pipeline, schema


def predict_success(model_dir: Path | None, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    pipeline, schema = load_model(model_dir)
    frame = pd.DataFrame(rows)
    missing = [column for column in schema["feature_columns"] if column not in frame.columns]
    if missing:
        raise ValueError("Missing inference features: " + ", ".join(sorted(missing)))

    frame = frame[schema["feature_columns"]]
    probabilities = pipeline.predict_proba(frame)[:, 1]
    return [
        {
            "rank": rank,
            "success_probability": float(probability),
            "features": {
                key: _json_safe(value) for key, value in rows[index].items()
            },
        }
        for rank, (index, probability) in enumerate(
            sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True),
            start=1,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--sample-row", type=Path, required=True)
    args = parser.parse_args()

    sample = json.loads(args.sample_row.read_text(encoding="utf-8"))
    predictions = predict_success(args.model_dir, [sample])
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()

