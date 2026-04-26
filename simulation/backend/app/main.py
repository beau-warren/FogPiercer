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

from app.candidate_actions import action_candidates, calibrated_probability, features_for_action
from app.sensor_mapping import build_base_features, weighted_power


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LHR_SRC = PROJECT_ROOT / "logit_hierarchical_regression" / "src"
sys.path.append(str(LHR_SRC))

from fogpiercer_lhr.inference import load_model  # noqa: E402


DEFAULT_MODEL_REPO = "fermsi/SCSP_fogpiercer-logit-hierarchical-regression"
DEFAULT_MERCURY_BASE_URL = "https://api.inceptionlabs.ai/v1"
DEFAULT_MERCURY_MODEL = "mercury-2"
DEMO_SECONDS = 5 * 60
LOG_DIR = PROJECT_ROOT / "simulation" / "logs"
MERCURY_SUMMARY_CACHE: dict[str, str] = {}


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
    is_alternate: bool = False
    applicability: float = 1.0


class DecisionSetResponse(BaseModel):
    decisions: list[DecisionResponse]


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
    friendly_weighted_power = weighted_power(friendlies)
    enemy_weighted_power = weighted_power(enemies)
    enemy_drone_alive = any(unit.type == "UAS" for unit in enemies)
    friendly_drone_alive = any(unit.type == "UAS" for unit in friendlies)
    proximity = max(0.0, min(1.0, (260.0 - closest) / 260.0))
    ratio = max(0.0, min(1.7, enemy_weighted_power / max(1.0, friendly_weighted_power))) / 1.7
    threat = round((proximity * 0.68 + ratio * 0.32) * 100.0)
    force_balance_index = max(-3.0, min(3.0, (friendly_weighted_power - enemy_weighted_power) / 100.0))

    return {
        "friendly_count": float(len(friendlies)),
        "enemy_count": float(len(enemies)),
        "friendly_power": friendly_power,
        "enemy_power": enemy_power,
        "friendly_weighted_power": round(friendly_weighted_power, 2),
        "enemy_weighted_power": round(enemy_weighted_power, 2),
        "force_balance_index": round(force_balance_index, 2),
        "vip_health": vip.health if vip else 0.0,
        "closest_enemy": closest,
        "enemy_drone_alive": enemy_drone_alive,
        "friendly_drone_alive": friendly_drone_alive,
        "threat": threat,
    }


def base_feature_row(
    state: SimulationState,
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    elapsed_days = max(0.0, DEMO_SECONDS - state.seconds_remaining) / 86400.0
    return build_base_features(
        metrics=metrics,
        selected_decision_id=state.selected_decision_id,
        elapsed_days=elapsed_days,
    )


def raw_rows(decision: dict[str, Any], features: dict[str, Any], metrics: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = [
        ["decision_action", decision["title"]],
        ["success_probability", f"{decision['success_probability']:.3f} ({decision['score']:.0f}%)"],
        ["raw_logit_probability", f"{decision.get('raw_logit_probability', decision['success_probability']):.6f}"],
        ["force_balance_multiplier", decision.get("force_balance_multiplier", 1.0)],
        ["action_fit_multiplier", decision.get("action_fit_multiplier", 1.0)],
        ["probability_formula", "raw_logit_probability * force_balance_multiplier * action_fit_multiplier"],
        ["model_source", decision["model_source"]],
        ["mercury_used", decision["mercury_used"]],
        ["mercury_summary", decision["mercury_summary"]],
        ["is_alternate", decision.get("is_alternate", False)],
        ["applicability", decision.get("applicability", 1.0)],
        ["active_sensor_conditions", ", ".join(decision.get("active_sensor_conditions", []))],
        ["target_column", "attacker_success"],
    ]
    for mapping in decision.get("sensor_mapping_trace", []):
        rows.append(
            [
                f"sensor_mapping.{mapping['condition']}",
                f"{mapping['description']} -> {mapping['fields']}",
            ]
        )
    rows.append(["action_feature_deltas", decision.get("action_deltas", {})])
    rows.extend([key, value] for key, value in features.items())
    rows.extend(
        [
            ["closest_enemy_m", round(float(metrics["closest_enemy"]) * 2.5)],
            ["friendly_combat_power", round(float(metrics["friendly_power"]))],
            ["enemy_combat_power", round(float(metrics["enemy_power"]))],
            ["friendly_weighted_power", round(float(metrics["friendly_weighted_power"]))],
            ["enemy_weighted_power", round(float(metrics["enemy_weighted_power"]))],
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
                "raw_logit_probability": decision.get("raw_logit_probability"),
                "force_balance_multiplier": decision.get("force_balance_multiplier"),
                "action_fit_multiplier": decision.get("action_fit_multiplier"),
                "model_source": decision["model_source"],
                "mercury_used": decision.get("mercury_used", False),
                "mercury_summary": decision.get("mercury_summary", ""),
                "is_alternate": decision.get("is_alternate", False),
                "applicability": decision.get("applicability", 1.0),
                "active_sensor_conditions": decision.get("active_sensor_conditions", []),
                "sensor_mapping_trace": decision.get("sensor_mapping_trace", []),
                "action_deltas": decision.get("action_deltas", {}),
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
        friendly_alive = sum(1 for unit in event.units if unit.side == "friendly" and unit.health > 0)
        enemy_alive = sum(1 for unit in event.units if unit.side == "enemy" and unit.health > 0)
        friendly_total = sum(1 for unit in event.units if unit.side == "friendly")
        enemy_total = sum(1 for unit in event.units if unit.side == "enemy")
        vip_health = next((unit.health for unit in event.units if unit.id == "vip"), 0)
        summary = {
            "timestamp": payload["timestamp"],
            "run_id": event.run_id,
            "event_type": event.event_type,
            "mission_status": event.mission_status,
            "selected_decision_id": event.selected_decision_id,
            "ticks": event.tick,
            "seconds_elapsed": round(DEMO_SECONDS - event.seconds_remaining, 2),
            "vip_killed": vip_health <= 0,
            "all_friendlies_eliminated": friendly_alive == 0,
            "all_enemies_eliminated": enemy_alive == 0,
            "friendly_alive": friendly_alive,
            "friendly_total": friendly_total,
            "enemy_alive": enemy_alive,
            "enemy_total": enemy_total,
            "vip_health": vip_health,
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
            "base_summary": decision["summary"],
            "success_probability": round(float(decision["success_probability"]), 3),
            "score": decision["score"],
            "is_alternate": decision.get("is_alternate", False),
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
                    "Preserve each option's distinct role and do not make all options sound alike. "
                    "Do not invent new options. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": (
                            "For each option, produce one clear sentence under 28 words. "
                            "Mention the distinct operational effect, not implementation details. "
                            "If an option is marked alternate, describe why it is secondary."
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

    missing: list[dict[str, Any]] = []
    for decision in decisions:
        summary_token = re.sub(r"[^A-Za-z0-9]+", "-", decision["summary"]).strip("-")[:48]
        cache_key = f"{decision['id']}:{round(float(decision['score']))}:{summary_token}"
        decision["mercury_cache_key"] = cache_key
        if cache_key in MERCURY_SUMMARY_CACHE:
            decision["mercury_summary"] = MERCURY_SUMMARY_CACHE[cache_key]
            decision["mercury_used"] = True
        else:
            missing.append(decision)

    if not missing:
        return decisions

    try:
        response = httpx.post(
            mercury_endpoint(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=mercury_payload(missing, metrics),
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
        for decision in missing:
            decision["mercury_summary"] = summaries.get(
                decision["id"],
                fallback_mercury_summary(decision, metrics),
            )
            decision["mercury_used"] = decision["id"] in summaries
            MERCURY_SUMMARY_CACHE[decision["mercury_cache_key"]] = decision["mercury_summary"]
    except Exception:
        for decision in missing:
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
    base_features, mapping_trace, sensor_conditions = base_feature_row(state, metrics)
    active_conditions = [condition for condition, active in sensor_conditions.items() if active]
    candidates = action_candidates(metrics)
    rows = [features_for_action(base_features, candidate) for candidate in candidates]

    import pandas as pd

    frame = pd.DataFrame(rows)[schema["feature_columns"]]
    probabilities = pipeline.predict_proba(frame)[:, 1]

    ranked_decisions: list[dict[str, Any]] = []
    for candidate, probability, features in zip(candidates, probabilities, rows, strict=True):
        applicability = float(candidate["applicability"])
        calibration = calibrated_probability(float(probability), candidate["id"], metrics)
        adjusted_probability = calibration["adjusted_probability"]
        ranked_decisions.append(
            {
                "id": candidate["id"],
                "title": candidate["title"],
                "summary": candidate["summary"],
                "features": features,
                "success_probability": adjusted_probability,
                "raw_logit_probability": float(probability),
                "force_balance_multiplier": calibration["force_balance_multiplier"],
                "action_fit_multiplier": calibration["action_fit_multiplier"],
                "score": round(adjusted_probability * 100.0, 1),
                "model_source": os.environ.get("FOGPIERCER_HF_MODEL_REPO", DEFAULT_MODEL_REPO),
                "is_alternate": False,
                "applicability": applicability,
                "active_sensor_conditions": active_conditions,
                "sensor_mapping_trace": mapping_trace,
                "action_deltas": candidate["deltas"],
            }
        )

    ranked_decisions.sort(key=lambda item: item["success_probability"], reverse=True)
    for index, decision in enumerate(ranked_decisions):
        decision["is_alternate"] = index >= 3
    ranked_decisions = enrich_with_mercury(ranked_decisions, metrics)

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
                is_alternate=decision["is_alternate"],
                applicability=decision["applicability"],
            )
        )

    log_decision_snapshot(state, ranked_decisions, metrics)
    return DecisionSetResponse(decisions=ranked)

