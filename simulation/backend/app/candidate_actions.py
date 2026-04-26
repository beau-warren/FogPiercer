from __future__ import annotations

from typing import Any


PRIMARY_ACTION_IDS = ["counter-uas", "break-contact", "screen-and-push"]

ACTION_CATALOG: dict[str, dict[str, Any]] = {
    "counter-uas": {
        "title": "Prioritize FPV drone intercept",
        "summary": "Commit friendly FPV drone and escort fires against the hostile drone before it reaches the VIP.",
        "posture": {"tactical_posture": "HD|postype_0", "postype": 0, "post1": "HD", "post2": None},
        "deltas": {"techa": 1, "intela": 1, "aeroa": 1, "aira": 1},
        "primary": True,
    },
    "break-contact": {
        "title": "Break contact and extract VIP",
        "summary": "Pull the VIP vehicle back through the cleared road segment while escorts cover the withdrawal.",
        "posture": {"tactical_posture": "PD|postype_0", "postype": 0, "post1": "PD", "post2": None},
        "deltas": {"resa": 1, "mobila": -1, "inita": -1, "direction": -1, "momnta": -1},
        "primary": True,
    },
    "screen-and-push": {
        "title": "Dismount screen and push through",
        "summary": "Use infantry and MRAP to screen the danger area while the VIP vehicle accelerates.",
        "posture": {"tactical_posture": "HD|postype_1", "postype": 1, "post1": "HD", "post2": "PD"},
        "deltas": {"mobila": 1, "inita": 1, "plana": 1, "momnta": 1},
        "primary": True,
    },
    "hold-defensive-perimeter": {
        "title": "Hold defensive perimeter",
        "summary": "Freeze the VIP escort, consolidate fields of fire, and force enemy cells to close into covered sectors.",
        "posture": {"tactical_posture": "HD|postype_0", "postype": 0, "post1": "HD", "post2": None},
        "deltas": {"fortsa": 1, "plana": 1, "momnta": -1, "mobila": -1},
        "primary": False,
    },
    "shift-vip-to-cover": {
        "title": "Shift VIP vehicle to cover",
        "summary": "Move the VIP platform off the exposed road while escort elements suppress the closest threat.",
        "posture": {"tactical_posture": "PD|postype_1", "postype": 1, "post1": "PD", "post2": "HD"},
        "deltas": {"resa": 1, "mobila": 1, "plana": 1, "quala": 1.4},
        "primary": False,
    },
    "call-for-reinforcement": {
        "title": "Call for reinforcement",
        "summary": "Delay decisive movement and request external support while preserving the current defensive posture.",
        "posture": {"tactical_posture": "HD|postype_0", "postype": 0, "post1": "HD", "post2": None},
        "deltas": {"logsa": 1, "logsaa": 1, "mana": 1, "inita": -1},
        "primary": False,
    },
}


def action_candidates(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for action_id, action in ACTION_CATALOG.items():
        candidate = dict(action)
        candidate["id"] = action_id
        candidate["applicability"] = applicability(action_id, metrics)
        if action_id == "counter-uas" and not bool(metrics["enemy_drone_alive"]):
            candidate["summary"] = "No hostile FPV drone is active; preserve friendly drone capacity for ground threats."
        candidates.append(candidate)
    return candidates


def applicability(action_id: str, metrics: dict[str, Any]) -> float:
    enemy_uas_alive = bool(metrics["enemy_drone_alive"])
    friendly_count = float(metrics["friendly_count"])
    enemy_count = float(metrics["enemy_count"])
    friendly_power = float(metrics["friendly_weighted_power"])
    enemy_power = float(metrics["enemy_weighted_power"])
    vip_health = float(metrics["vip_health"])

    if action_id == "counter-uas":
        return 1.0 if enemy_uas_alive else 0.10
    if action_id == "screen-and-push":
        if enemy_count >= max(2.0, friendly_count * 2.0):
            return 0.45
        return 0.9 if friendly_power >= enemy_power * 0.8 else 0.55
    if action_id == "break-contact":
        return 0.95 if enemy_power > friendly_power or vip_health < 70 else 0.75
    if action_id == "shift-vip-to-cover":
        return 0.9 if vip_health < 80 or enemy_power > friendly_power else 0.65
    if action_id == "hold-defensive-perimeter":
        return 0.85 if enemy_count > friendly_count else 0.55
    if action_id == "call-for-reinforcement":
        return 0.8 if enemy_power > friendly_power * 1.25 else 0.45
    return 0.5


def force_balance_multiplier(metrics: dict[str, Any]) -> float:
    friendly_power = float(metrics["friendly_weighted_power"])
    enemy_power = float(metrics["enemy_weighted_power"])
    friendly_count = float(metrics["friendly_count"])
    enemy_count = float(metrics["enemy_count"])

    if enemy_power <= 0:
        return 1.15
    if friendly_power <= 0:
        return 0.05

    power_ratio = friendly_power / enemy_power
    count_ratio = friendly_count / max(1.0, enemy_count)
    blended_ratio = (power_ratio * 0.75) + (count_ratio * 0.25)
    return round(max(0.05, min(1.15, blended_ratio**0.45)), 4)


def calibrated_probability(
    raw_logit_probability: float,
    action_id: str,
    metrics: dict[str, Any],
) -> dict[str, float]:
    force_multiplier = force_balance_multiplier(metrics)
    action_multiplier = applicability(action_id, metrics)
    adjusted = float(raw_logit_probability) * force_multiplier * action_multiplier
    return {
        "adjusted_probability": round(max(0.01, min(0.99, adjusted)), 4),
        "force_balance_multiplier": force_multiplier,
        "action_fit_multiplier": round(action_multiplier, 4),
    }


def features_for_action(base_features: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    row = dict(base_features)
    row.update(action["posture"])
    row.update(action["deltas"])
    return row

