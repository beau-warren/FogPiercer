from __future__ import annotations

from typing import Any


TYPE_WEIGHTS = {
    "VIP": 0.35,
    "UAS": 0.65,
    "INF": 1.0,
    "MRAP": 1.45,
}

BASE_CDB90_FEATURES: dict[str, Any] = {
    "war4_theater": "Modern local demo",
    "terrain_primary": "R",
    "weather_primary": "D",
    "primary_attacker": "Friendly convoy",
    "primary_defender": "Hostile ambush cell",
    "front": 0,
    "depth": 0,
    "aeroa": 0,
    "surpa": 0,
    "cea": 0,
    "leada": 0,
    "trnga": 0,
    "morala": 0,
    "logsa": 0,
    "momnta": 0,
    "intela": 0,
    "techa": 0,
    "inita": 0,
    "quala": 1.0,
    "resa": 0,
    "mobila": 0,
    "aira": 0,
    "fprepa": 0,
    "wxa": 0,
    "terra": 1,
    "leadaa": 0,
    "plana": 0,
    "surpaa": 0,
    "mana": 0,
    "logsaa": 0,
    "fortsa": 0,
    "deepa": 0,
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
    "duration2": 0.0,
    "dyad_weight": 1.0,
    "direction": 1,
}

CONDITION_MAP: dict[str, dict[str, Any]] = {
    "friendly_badly_outnumbered": {
        "description": "Enemy alive count is at least twice friendly alive count.",
        "fields": {"cea": -2, "morala": -1, "momnta": -1, "surpa": -1},
    },
    "friendly_outnumber_enemy": {
        "description": "Friendly alive count is materially higher than enemy alive count.",
        "fields": {"cea": 1, "morala": 1, "momnta": 1},
    },
    "enemy_power_advantage": {
        "description": "Enemy weighted combat power exceeds friendly weighted combat power.",
        "fields": {"cea": -1, "front": 1, "depth": 1, "deepa": 1},
    },
    "friendly_power_advantage": {
        "description": "Friendly weighted combat power exceeds enemy weighted combat power.",
        "fields": {"cea": 1, "inita": 1, "momnta": 1},
    },
    "enemy_uas_alive": {
        "description": "Hostile UAS is active and can coordinate the ambush.",
        "fields": {"aeroa": -1, "aira": -1, "intela": 0, "techa": -1},
    },
    "no_enemy_uas": {
        "description": "No hostile UAS remains, reducing the value of counter-UAS action.",
        "fields": {"aeroa": 1, "aira": 0, "intela": 1, "techa": 0},
    },
    "friendly_uas_alive": {
        "description": "Friendly ISR UAS is alive and improving observation.",
        "fields": {"intela": 1, "techa": 1},
    },
    "enemy_in_close_contact": {
        "description": "Enemy is inside close weapon range.",
        "fields": {"front": 1, "depth": 1, "surpa": -1, "deepa": 1},
    },
    "vip_degraded": {
        "description": "VIP platform has taken meaningful damage.",
        "fields": {"quala": 0.6, "morala": -1, "resa": 1},
    },
    "vip_healthy": {
        "description": "VIP platform remains healthy.",
        "fields": {"quala": 2.0, "morala": 1},
    },
    "commander_decision_active": {
        "description": "A commander option is active.",
        "fields": {"leada": 1, "leadaa": 1, "plana": 1},
    },
}


def weighted_power(units: list[Any]) -> float:
    total = 0.0
    for unit in units:
        total += float(unit.health) * TYPE_WEIGHTS.get(unit.type, 1.0)
    return total


def derive_conditions(metrics: dict[str, Any], selected_decision_id: str | None) -> dict[str, bool]:
    friendly_count = float(metrics["friendly_count"])
    enemy_count = float(metrics["enemy_count"])
    friendly_weighted_power = float(metrics["friendly_weighted_power"])
    enemy_weighted_power = float(metrics["enemy_weighted_power"])
    return {
        "friendly_badly_outnumbered": enemy_count >= max(2.0, friendly_count * 2.0),
        "friendly_outnumber_enemy": friendly_count >= enemy_count + 2.0,
        "enemy_power_advantage": enemy_weighted_power > friendly_weighted_power * 1.2,
        "friendly_power_advantage": friendly_weighted_power > enemy_weighted_power * 1.2,
        "enemy_uas_alive": bool(metrics["enemy_drone_alive"]),
        "no_enemy_uas": not bool(metrics["enemy_drone_alive"]),
        "friendly_uas_alive": bool(metrics["friendly_drone_alive"]),
        "enemy_in_close_contact": float(metrics["closest_enemy"]) <= 125.0,
        "vip_degraded": float(metrics["vip_health"]) < 65.0,
        "vip_healthy": float(metrics["vip_health"]) >= 80.0,
        "commander_decision_active": bool(selected_decision_id),
    }


def build_base_features(
    *,
    metrics: dict[str, Any],
    selected_decision_id: str | None,
    elapsed_days: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    features = dict(BASE_CDB90_FEATURES)
    features["duration2"] = round(elapsed_days, 5)
    features["cea"] = round(float(metrics["force_balance_index"]), 2)
    features["quala"] = round(float(metrics["vip_health"]) / 50.0, 2)

    conditions = derive_conditions(metrics, selected_decision_id)
    trace: list[dict[str, Any]] = []
    for condition, active in conditions.items():
        if not active:
            continue
        mapping = CONDITION_MAP[condition]
        features.update(mapping["fields"])
        trace.append(
            {
                "condition": condition,
                "description": mapping["description"],
                "fields": mapping["fields"],
            }
        )

    return features, trace, conditions

