from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LHR_SRC = PROJECT_ROOT / "logit_hierarchical_regression" / "src"
sys.path.append(str(LHR_SRC))

from fogpiercer_lhr.inference import load_model  # noqa: E402


DEFAULT_MODEL_REPO = "fermsi/SCSP_fogpiercer-logit-hierarchical-regression"
DEMO_SECONDS = 5 * 60


class UnitState(BaseModel):
    id: str
    side: str
    type: str
    label: str
    x: float
    y: float
    health: float
    range: float


class SimulationState(BaseModel):
    seconds_remaining: float = Field(alias="secondsRemaining")
    selected_decision_id: str | None = Field(default=None, alias="selectedDecisionId")
    units: list[UnitState]


class DecisionResponse(BaseModel):
    id: str
    title: str
    summary: str
    score: float
    success_probability: float
    model_source: str
    raw_rows: list[list[Any]]
    features: dict[str, Any]


class DecisionSetResponse(BaseModel):
    decisions: list[DecisionResponse]


DECISION_TEMPLATES = {
    "counter-uas": {
        "title": "Prioritize counter-UAS intercept",
        "summary": "Move ISR and escort fires onto the hostile drone to reduce ambush coordination.",
        "posture": "HD|postype_0",
        "post1": "HD",
        "post2": None,
        "postype": 0,
    },
    "break-contact": {
        "title": "Break contact and reverse convoy",
        "summary": "Pull the VIP vehicle back through the cleared road segment while escorts cover.",
        "posture": "PD|postype_0",
        "post1": "PD",
        "post2": None,
        "postype": 0,
    },
    "screen-and-push": {
        "title": "Dismount screen and push through",
        "summary": "Use infantry and MRAP to screen the danger area while the VIP vehicle accelerates.",
        "posture": "HD|postype_1",
        "post1": "HD",
        "post2": "PD",
        "postype": 1,
    },
}


app = FastAPI(title="SCSP Fogpiercer Simulation Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def configure_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("FOGPIERCER_HF_MODEL_REPO", DEFAULT_MODEL_REPO)
    os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(PROJECT_ROOT / ".hf_cache" / "xet"))
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_XET_CACHE"]).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_model() -> tuple[object, dict[str, Any]]:
    configure_environment()
    return load_model(model_dir=None)


def live_units(state: SimulationState, side: str | None = None) -> list[UnitState]:
    units = [unit for unit in state.units if unit.health > 0]
    if side:
        return [unit for unit in units if unit.side == side]
    return units


def euclidean(a: UnitState, b: UnitState) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def battlefield_metrics(state: SimulationState) -> dict[str, float | bool]:
    friendlies = live_units(state, "friendly")
    enemies = live_units(state, "enemy")
    vip = next((unit for unit in state.units if unit.id == "vip"), None)
    closest = min(
        [euclidean(enemy, friendly) for enemy in enemies for friendly in friendlies],
        default=999.0,
    )
    friendly_power = sum(unit.health for unit in friendlies)
    enemy_power = sum(unit.health for unit in enemies)
    enemy_drone_alive = any(unit.type == "UAS" for unit in enemies)
    proximity = max(0.0, min(1.0, (260.0 - closest) / 260.0))
    ratio = max(0.0, min(1.7, enemy_power / max(1.0, friendly_power))) / 1.7
    threat = round((proximity * 0.68 + ratio * 0.32) * 100.0)

    return {
        "friendly_count": float(len(friendlies)),
        "enemy_count": float(len(enemies)),
        "friendly_power": friendly_power,
        "enemy_power": enemy_power,
        "vip_health": vip.health if vip else 0.0,
        "closest_enemy": closest,
        "enemy_drone_alive": enemy_drone_alive,
        "threat": threat,
    }


def base_feature_row(state: SimulationState) -> dict[str, Any]:
    metrics = battlefield_metrics(state)
    threat = float(metrics["threat"])
    enemy_drone_alive = bool(metrics["enemy_drone_alive"])
    elapsed_days = max(0.0, DEMO_SECONDS - state.seconds_remaining) / 86400.0

    return {
        "war4_theater": "Modern local demo",
        "terrain_primary": "R",
        "weather_primary": "D",
        "primary_attacker": "Friendly convoy",
        "primary_defender": "Hostile ambush cell",
        "front": 1 if threat > 45 else 0,
        "depth": 1 if threat > 65 else 0,
        "aeroa": -1 if enemy_drone_alive else 1,
        "surpa": -1 if threat > 55 else 0,
        "cea": round((float(metrics["friendly_count"]) - float(metrics["enemy_count"])) * 0.5, 2),
        "leada": 1 if state.selected_decision_id else 0,
        "trnga": 0,
        "morala": 1 if float(metrics["vip_health"]) > 70 else 0,
        "logsa": 0,
        "momnta": 1 if threat < 45 else -1,
        "intela": 0 if enemy_drone_alive else 1,
        "techa": 0,
        "inita": 0,
        "quala": round(float(metrics["vip_health"]) / 50.0, 2),
        "resa": 0,
        "mobila": 0,
        "aira": -1 if enemy_drone_alive else 0,
        "fprepa": 0,
        "wxa": 0,
        "terra": 1,
        "leadaa": 1 if state.selected_decision_id else 0,
        "plana": 0,
        "surpaa": -1 if threat > 55 else 0,
        "mana": 0,
        "logsaa": 0,
        "fortsa": 0,
        "deepa": 1 if threat > 65 else 0,
        "is_hero": 1,
        "war_initiator": 1,
        "terrano": 1,
        "terra1": "R",
        "terra2": "M",
        "terra3": None,
        "wxno": 1,
        "wx1": "D",
        "wx2": "S",
        "wx3": "T",
        "wx4": "F",
        "wx5": "T",
        "duration1": 1.0,
        "duration2": round(elapsed_days, 5),
        "dyad_weight": 1.0,
        "direction": 1,
    }


def feature_row_for_decision(state: SimulationState, decision_id: str) -> dict[str, Any]:
    template = DECISION_TEMPLATES[decision_id]
    row = base_feature_row(state)
    row.update(
        {
            "tactical_posture": template["posture"],
            "postype": template["postype"],
            "post1": template["post1"],
            "post2": template["post2"],
        }
    )

    if decision_id == "counter-uas":
        row.update({"techa": 1, "intela": 1, "aeroa": 1})
    elif decision_id == "break-contact":
        row.update({"resa": 1, "mobila": -1, "inita": -1, "direction": -1})
    elif decision_id == "screen-and-push":
        row.update({"mobila": 1, "inita": 1, "plana": 1, "momnta": 1})

    return row


def raw_rows(decision: dict[str, Any], features: dict[str, Any], metrics: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = [
        ["decision_action", decision["title"]],
        ["success_probability", f"{decision['success_probability']:.3f} ({decision['score']:.0f}%)"],
        ["model_source", decision["model_source"]],
        ["target_column", "attacker_success"],
    ]
    rows.extend([key, value] for key, value in features.items())
    rows.extend(
        [
            ["closest_enemy_m", round(float(metrics["closest_enemy"]) * 2.5)],
            ["friendly_combat_power", round(float(metrics["friendly_power"]))],
            ["enemy_combat_power", round(float(metrics["enemy_power"]))],
            ["threat_index", round(float(metrics["threat"]))],
        ]
    )
    return rows


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/decisions", response_model=DecisionSetResponse)
def decisions(state: SimulationState) -> DecisionSetResponse:
    pipeline, schema = get_model()
    metrics = battlefield_metrics(state)
    rows = [feature_row_for_decision(state, decision_id) for decision_id in DECISION_TEMPLATES]

    import pandas as pd

    frame = pd.DataFrame(rows)[schema["feature_columns"]]
    probabilities = pipeline.predict_proba(frame)[:, 1]

    ranked: list[DecisionResponse] = []
    for decision_id, probability, features in zip(DECISION_TEMPLATES, probabilities, rows, strict=True):
        template = DECISION_TEMPLATES[decision_id]
        decision = {
            "title": template["title"],
            "success_probability": float(probability),
            "score": round(float(probability) * 100.0, 1),
            "model_source": os.environ.get("FOGPIERCER_HF_MODEL_REPO", DEFAULT_MODEL_REPO),
        }
        ranked.append(
            DecisionResponse(
                id=decision_id,
                title=template["title"],
                summary=template["summary"],
                score=decision["score"],
                success_probability=decision["success_probability"],
                model_source=decision["model_source"],
                raw_rows=raw_rows(decision, features, metrics),
                features=features,
            )
        )

    ranked.sort(key=lambda item: item.success_probability, reverse=True)
    return DecisionSetResponse(decisions=ranked[:3])

