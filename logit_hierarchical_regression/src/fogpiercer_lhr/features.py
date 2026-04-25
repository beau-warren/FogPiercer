from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FeatureSchema:
    target_column: str
    action_column: str
    group_columns: list[str]
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "target_column": self.target_column,
            "action_column": self.action_column,
            "group_columns": self.group_columns,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
        }


def build_feature_schema(
    frame: pd.DataFrame,
    target_column: str,
    action_column: str,
    group_columns: Iterable[str],
) -> FeatureSchema:
    required = [target_column, action_column, *group_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Training config references missing CDB90 columns: "
            + ", ".join(sorted(missing))
        )

    group_columns = list(group_columns)
    ignored_columns = {target_column}
    feature_columns = [column for column in frame.columns if column not in ignored_columns]

    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]

    # Ensure action and hierarchy columns are categorical even if encoded as ints.
    for column in [action_column, *group_columns]:
        if column in numeric_columns:
            numeric_columns.remove(column)
        if column not in categorical_columns:
            categorical_columns.append(column)

    return FeatureSchema(
        target_column=target_column,
        action_column=action_column,
        group_columns=group_columns,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def normalize_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    lowered = series.astype(str).str.lower().str.strip()
    positive = {"1", "true", "success", "win", "yes", "y"}
    negative = {"0", "false", "failure", "fail", "loss", "no", "n"}

    if lowered.isin(positive | negative).all():
        return lowered.map(lambda value: 1 if value in positive else 0).astype(int)

    if pd.api.types.is_numeric_dtype(series):
        unique = set(series.dropna().unique().tolist())
        if unique <= {0, 1}:
            return series.astype(int)

    raise ValueError(
        "Target column must be binary. Configure target_column to a CDB90 success/outcome field."
    )

