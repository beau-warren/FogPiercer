from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fogpiercer_lhr.config import PROJECT_ROOT


CDB90_COLUMNS = [
    "isqno",
    "war",
    "name",
    "campgn",
    "postype",
    "post1",
    "post2",
    "front",
    "depth",
    "aeroa",
    "surpa",
    "cea",
    "leada",
    "trnga",
    "morala",
    "logsa",
    "momnta",
    "intela",
    "techa",
    "inita",
    "quala",
    "resa",
    "mobila",
    "aira",
    "fprepa",
    "wxa",
    "terra",
    "leadaa",
    "plana",
    "surpaa",
    "mana",
    "logsaa",
    "fortsa",
    "deepa",
    "is_hero",
    "war4_theater",
    "war_initiator",
]


def _first_present(values: pd.Series, fallback: str = "unknown") -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value)
    return fallback


def _tactical_posture(row: pd.Series) -> str:
    primary = _first_present(pd.Series([row.get("post1"), row.get("post2")]))
    posture_type = row.get("postype")
    return f"{primary}|postype_{posture_type if pd.notna(posture_type) else 'unknown'}"


def build_model_table(cdb90_root: Path) -> pd.DataFrame:
    data_dir = cdb90_root / "data"
    battles = pd.read_csv(data_dir / "battles.csv")
    terrain = pd.read_csv(data_dir / "terrain.csv")
    weather = pd.read_csv(data_dir / "weather.csv")
    durations = pd.read_csv(data_dir / "battle_durations.csv")
    dyads = pd.read_csv(data_dir / "battle_dyads.csv")

    primary_dyads = (
        dyads[dyads["primary"].astype(bool)]
        .loc[:, ["isqno", "attacker", "defender", "wt", "direction"]]
        .rename(
            columns={
                "attacker": "primary_attacker",
                "defender": "primary_defender",
                "wt": "dyad_weight",
            }
        )
    )

    table = (
        battles.loc[:, [column for column in CDB90_COLUMNS + ["wina"] if column in battles.columns]]
        .merge(terrain, on="isqno", how="left")
        .merge(weather, on="isqno", how="left")
        .merge(durations.loc[:, ["isqno", "duration1", "duration2"]], on="isqno", how="left")
        .merge(primary_dyads, on="isqno", how="left")
    )

    table["attacker_success"] = (table["wina"] == 1).astype(int)
    table["terrain_primary"] = table["terra1"].fillna("unknown")
    table["weather_primary"] = table["wx1"].fillna("unknown")
    table["tactical_posture"] = table.apply(_tactical_posture, axis=1)
    table["had_draw_or_loss"] = np.where(table["wina"].isin([0, -1]), 1, 0)

    # Drop leakage fields and free-text identifiers that should not drive demo scoring.
    leakage_or_ids = {
        "wina",
        "had_draw_or_loss",
        "isqno",
        "name",
        "campgn",
        "war",
    }
    ordered_columns = [
        "attacker_success",
        "tactical_posture",
        "war4_theater",
        "terrain_primary",
        "weather_primary",
        "primary_attacker",
        "primary_defender",
        *[
            column
            for column in table.columns
            if column not in leakage_or_ids
            and column
            not in {
                "attacker_success",
                "tactical_posture",
                "war4_theater",
                "terrain_primary",
                "weather_primary",
                "primary_attacker",
                "primary_defender",
            }
        ],
    ]
    return table.loc[:, ordered_columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cdb90-root",
        type=Path,
        default=PROJECT_ROOT / "logit_hierarchical_regression" / "data" / "raw" / "CDB90",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "logit_hierarchical_regression"
        / "data"
        / "processed"
        / "cdb90_model_table.csv",
    )
    args = parser.parse_args()

    table = build_model_table(args.cdb90_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table)} rows and {len(table.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()

