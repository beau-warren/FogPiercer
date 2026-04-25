from __future__ import annotations

import os
import sys
import json
import csv
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LHR_SRC = PROJECT_ROOT / "logit_hierarchical_regression" / "src"
sys.path.append(str(LHR_SRC))

from fogpiercer_lhr.inference import load_model  # noqa: E402


DEFAULT_MODEL_REPO = "fermsi/SCSP_fogpiercer-logit-hierarchical-regression"
DEFAULT_MERCURY_BASE_URL = "https://api.inceptionlabs.ai/v1"
DEFAULT_MERCURY_MODEL = "mercury-2"
DEMO_SECONDS = 5 * 60
LOG_DIR = PROJECT_ROOT / "simulation" / "logs"


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
    run_id: str = Field(default="manual-run", alias="runId")
    tick: int = 0
    seconds_remaining: float = Field(alias="secondsRemaining")
    selected_decision_id: str | None = Field(default=None, alias="selectedDecisionId")
    units: list[UnitState]


class SimulationEvent(BaseModel):
    run_id: str = Field(alias="runId")
    event_type: str = Field(alias="eventType")
    tick: int
    seconds_remaining: float = Field(alias="secondsRemaining")
    selected_decision_id: str | None = Field(default=None, alias="selectedDecisionId")
    mission_status: str = Field(default="running", alias="missionStatus")
    units: list[UnitState]


class DecisionResponse(BaseModel):
    id: str
    title: str
    summary: str
    score: float
    success_probability: float
    model_source: str
    mercury_summary: str
    mercury_used: bool
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
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    os.environ.setdefault("FOGPIERCER_HF_MODEL_REPO", DEFAULT_MODEL_REPO)
    os.environ.setdefault("MERCURY_II_BASE_URL", DEFAULT_MERCURY_BASE_URL)
    os.environ.setdefault("MERCURY_II_MODEL", DEFAULT_MERCURY_MODEL)
    os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(PROJECT_ROOT / ".hf_cache" / "xet"))
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_XET_CACHE"]).mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_run_id(run_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip("-")
    return cleaned[:80] or "manual-run"


def run_log_dir(run_id: str) -> Path:
    path = LOG_DIR / safe_run_id(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, default=str) + "\n")


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
        ["mercury_used", decision["mercury_used"]],
        ["mercury_summary", decision["mercury_summary"]],
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


def log_decision_snapshot(
    state: SimulationState,
    decisions: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    directory = run_log_dir(state.run_id)
    timestamp = utc_now()
    snapshot = {
        "timestamp": timestamp,
        "run_id": state.run_id,
        "tick": state.tick,
        "seconds_remaining": state.seconds_remaining,
        "selected_decision_id": state.selected_decision_id,
        "metrics": metrics,
        "units": [unit.model_dump() for unit in state.units],
        "decisions": [
            {
                "id": decision["id"],
                "title": decision["title"],
                "score": decision["score"],
                "success_probability": decision["success_probability"],
                "model_source": decision["model_source"],
                "mercury_used": decision.get("mercury_used", False),
                "mercury_summary": decision.get("mercury_summary", ""),
                "features": decision["features"],
            }
            for decision in decisions
        ],
    }
    append_jsonl(directory / "decision_snapshots.jsonl", snapshot)
    append_training_rows(directory / "cdb90_training_rows.csv", timestamp, state, decisions, metrics)


def append_training_rows(
    path: Path,
    timestamp: str,
    state: SimulationState,
    decisions: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    feature_keys = sorted({key for decision in decisions for key in decision["features"]})
    fieldnames = [
        "timestamp",
        "run_id",
        "tick",
        "seconds_remaining",
        "option_id",
        "option_title",
        "selected_decision_id",
        "was_selected",
        "success_probability",
        "model_source",
        "mercury_used",
        "threat_index",
        "friendly_combat_power",
        "enemy_combat_power",
        "closest_enemy_m",
        "observed_outcome_pending",
        *feature_keys,
    ]
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for decision in decisions:
            row = {
                "timestamp": timestamp,
                "run_id": state.run_id,
                "tick": state.tick,
                "seconds_remaining": round(state.seconds_remaining, 2),
                "option_id": decision["id"],
                "option_title": decision["title"],
                "selected_decision_id": state.selected_decision_id or "",
                "was_selected": decision["id"] == state.selected_decision_id,
                "success_probability": decision["success_probability"],
                "model_source": decision["model_source"],
                "mercury_used": decision.get("mercury_used", False),
                "threat_index": metrics["threat"],
                "friendly_combat_power": round(float(metrics["friendly_power"]), 2),
                "enemy_combat_power": round(float(metrics["enemy_power"]), 2),
                "closest_enemy_m": round(float(metrics["closest_enemy"]) * 2.5),
                "observed_outcome_pending": True,
            }
            row.update(decision["features"])
            writer.writerow(row)


def log_simulation_event(event: SimulationEvent) -> None:
    directory = run_log_dir(event.run_id)
    payload = {
        "timestamp": utc_now(),
        "run_id": event.run_id,
        "event_type": event.event_type,
        "tick": event.tick,
        "seconds_remaining": event.seconds_remaining,
        "selected_decision_id": event.selected_decision_id,
        "mission_status": event.mission_status,
        "units": [unit.model_dump() for unit in event.units],
    }
    append_jsonl(directory / "events.jsonl", payload)

    if event.event_type in {"ended", "reset", "reiterate"}:
        summary = {
            "timestamp": payload["timestamp"],
            "run_id": event.run_id,
            "event_type": event.event_type,
            "mission_status": event.mission_status,
            "selected_decision_id": event.selected_decision_id,
            "ticks": event.tick,
            "seconds_elapsed": round(DEMO_SECONDS - event.seconds_remaining, 2),
            "friendly_alive": sum(1 for unit in event.units if unit.side == "friendly" and unit.health > 0),
            "enemy_alive": sum(1 for unit in event.units if unit.side == "enemy" and unit.health > 0),
            "vip_health": next((unit.health for unit in event.units if unit.id == "vip"), 0),
            "note": "Synthetic demo telemetry only. Use as candidate retraining data after human review; no automatic retraining is performed.",
        }
        (directory / "final_summary.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )


def mercury_endpoint() -> str:
    base_url = os.environ.get("MERCURY_II_BASE_URL", DEFAULT_MERCURY_BASE_URL).rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def fallback_mercury_summary(decision: dict[str, Any], metrics: dict[str, Any]) -> str:
    threat = round(float(metrics["threat"]))
    score = round(float(decision["score"]))
    return (
        f"{decision['title']} has {score}% model-estimated success. "
        f"Threat index is {threat}; act before hostile cells close range."
    )


def mercury_payload(decisions: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    compact_decisions = [
        {
            "id": decision["id"],
            "title": decision["title"],
            "success_probability": round(float(decision["success_probability"]), 3),
            "score": decision["score"],
        }
        for decision in decisions
    ]
    return {
        "model": os.environ.get("MERCURY_II_MODEL", DEFAULT_MERCURY_MODEL),
        "temperature": 0.2,
        "max_tokens": 700,
        "reasoning_effort": "low",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Mercury II supporting a tactical demo UI. "
                    "Write concise, non-classified, commander-facing option summaries. "
                    "Do not invent new options. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": (
                            "For each option, produce one clear sentence under 28 words. "
                            "Mention the likely operational effect, not implementation details."
                        ),
                        "battlefield_metrics": metrics,
                        "ranked_logit_options": compact_decisions,
                        "required_json_schema": {
                            "decisions": [
                                {
                                    "id": "same id from ranked_logit_options",
                                    "summary": "one concise sentence",
                                }
                            ]
                        },
                    }
                ),
            },
        ],
    }


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Mercury response did not include a JSON object")
    return json.loads(stripped[start : end + 1])


def enrich_with_mercury(decisions: list[dict[str, Any]], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    api_key = os.environ.get("MERCURY_II_API_KEY")
    if not api_key:
        for decision in decisions:
            decision["mercury_summary"] = fallback_mercury_summary(decision, metrics)
            decision["mercury_used"] = False
        return decisions

    try:
        response = httpx.post(
            mercury_endpoint(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=mercury_payload(decisions, metrics),
            timeout=8.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = extract_json_object(content)
        summaries = {
            item["id"]: item["summary"]
            for item in parsed.get("decisions", [])
            if item.get("id") and item.get("summary")
        }
        for decision in decisions:
            decision["mercury_summary"] = summaries.get(
                decision["id"],
                fallback_mercury_summary(decision, metrics),
            )
            decision["mercury_used"] = decision["id"] in summaries
    except Exception:
        for decision in decisions:
            decision["mercury_summary"] = fallback_mercury_summary(decision, metrics)
            decision["mercury_used"] = False

    return decisions


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/events")
def simulation_event(event: SimulationEvent) -> dict[str, str]:
    log_simulation_event(event)
    return {"status": "logged"}


@app.post("/api/decisions", response_model=DecisionSetResponse)
def decisions(state: SimulationState) -> DecisionSetResponse:
    pipeline, schema = get_model()
    metrics = battlefield_metrics(state)
    rows = [feature_row_for_decision(state, decision_id) for decision_id in DECISION_TEMPLATES]

    import pandas as pd

    frame = pd.DataFrame(rows)[schema["feature_columns"]]
    probabilities = pipeline.predict_proba(frame)[:, 1]

    ranked_decisions: list[dict[str, Any]] = []
    for decision_id, probability, features in zip(DECISION_TEMPLATES, probabilities, rows, strict=True):
        template = DECISION_TEMPLATES[decision_id]
        ranked_decisions.append(
            {
                "id": decision_id,
                "title": template["title"],
                "summary": template["summary"],
                "features": features,
                "success_probability": float(probability),
                "score": round(float(probability) * 100.0, 1),
                "model_source": os.environ.get("FOGPIERCER_HF_MODEL_REPO", DEFAULT_MODEL_REPO),
            }
        )

    ranked_decisions.sort(key=lambda item: item["success_probability"], reverse=True)
    ranked_decisions = enrich_with_mercury(ranked_decisions[:3], metrics)

    ranked: list[DecisionResponse] = []
    for decision in ranked_decisions:
        ranked.append(
            DecisionResponse(
                id=decision["id"],
                title=decision["title"],
                summary=decision["mercury_summary"],
                score=decision["score"],
                success_probability=decision["success_probability"],
                model_source=decision["model_source"],
                mercury_summary=decision["mercury_summary"],
                mercury_used=decision["mercury_used"],
                raw_rows=raw_rows(decision, decision["features"], metrics),
                features=decision["features"],
            )
        )

    log_decision_snapshot(state, ranked_decisions, metrics)
    return DecisionSetResponse(decisions=ranked)

