from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fogpiercer_lhr.config import PROJECT_ROOT, env_path, env_value, load_project_env, read_yaml
from fogpiercer_lhr.features import build_feature_schema, normalize_binary_target
from fogpiercer_lhr.hf_io import push_model_dir


def train(config_path: Path, push_to_hf: bool) -> dict[str, object]:
    load_project_env()
    config = read_yaml(config_path)
    csv_path = env_path(config["data"]["csv_path_env"])

    frame = pd.read_csv(csv_path)
    schema = build_feature_schema(
        frame=frame,
        target_column=config["data"]["target_column"],
        action_column=config["data"]["action_column"],
        group_columns=config["data"]["group_columns"],
    )

    y = normalize_binary_target(frame[schema.target_column])
    x = frame[schema.feature_columns].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                schema.numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                schema.categorical_columns,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=config["model"]["max_iter"],
                    class_weight=config["model"]["class_weight"],
                ),
            ),
        ]
    )

    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "rows": int(len(frame)),
        "feature_count": int(len(schema.feature_columns)),
    }

    output_dir = PROJECT_ROOT / "logit_hierarchical_regression" / config["artifacts"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, output_dir / config["artifacts"]["pipeline_file"])
    (output_dir / config["artifacts"]["feature_schema_file"]).write_text(
        json.dumps(schema.to_dict(), indent=2),
        encoding="utf-8",
    )
    (output_dir / config["artifacts"]["metadata_file"]).write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    sample_path = PROJECT_ROOT / "logit_hierarchical_regression" / config["artifacts"]["inference_sample_file"]
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample = x_test.iloc[0].where(pd.notna(x_test.iloc[0]), None).to_dict()
    sample_path.write_text(
        json.dumps(sample, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )

    if push_to_hf:
        repo_id = env_value(config["huggingface"]["repo_id_env"])
        token = env_value(config["huggingface"]["token_env"], required=False)
        push_model_dir(output_dir, repo_id=repo_id, token=token)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--push-to-hf", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    metrics = train(Path(args.config), push_to_hf=args.push_to_hf)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

