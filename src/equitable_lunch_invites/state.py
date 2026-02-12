from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from equitable_lunch_invites.models import RoleStats, STATE_SCHEMA_VERSION

ROLE_HOSTS = "hosts"
ROLE_GUESTS = "guests"
ROLE_NAMES = (ROLE_HOSTS, ROLE_GUESTS)


def default_state() -> dict[str, Any]:
    return {
        ROLE_HOSTS: {},
        ROLE_GUESTS: {},
        "_meta": {
            "schema_version": STATE_SCHEMA_VERSION,
            "event_history": {},
        },
    }


def _normalize_role_bucket(raw_bucket: Any) -> dict[str, dict[str, int]]:
    if not isinstance(raw_bucket, dict):
        return {}
    normalized: dict[str, dict[str, int]] = {}
    for name, raw_stats in raw_bucket.items():
        if not isinstance(name, str):
            continue
        stats = RoleStats.from_mapping(raw_stats if isinstance(raw_stats, dict) else {})
        normalized[name] = stats.to_mapping()
    return normalized


def migrate_state(raw_state: dict[str, Any] | None) -> dict[str, Any]:
    if not raw_state:
        return default_state()

    if ROLE_HOSTS in raw_state or ROLE_GUESTS in raw_state:
        state = default_state()
        state[ROLE_HOSTS] = _normalize_role_bucket(raw_state.get(ROLE_HOSTS))
        state[ROLE_GUESTS] = _normalize_role_bucket(raw_state.get(ROLE_GUESTS))
        meta = raw_state.get("_meta", {})
        if not isinstance(meta, dict):
            meta = {}
        merged_meta = state["_meta"]
        merged_meta.update(meta)
        merged_meta["schema_version"] = STATE_SCHEMA_VERSION
        merged_meta.setdefault("event_history", {})
        return state

    # Legacy migration from single-role top-level state:
    # { "<name>": {"attended_count": ...}, "_meta": {"cohort": [...], ...} }
    state = default_state()
    for name, raw_stats in raw_state.items():
        if name == "_meta" or not isinstance(name, str):
            continue
        if not isinstance(raw_stats, dict):
            raw_stats = {}
        state[ROLE_GUESTS][name] = RoleStats.from_mapping(raw_stats).to_mapping()

    legacy_meta = raw_state.get("_meta", {})
    if isinstance(legacy_meta, dict):
        if "cohort" in legacy_meta:
            state["_meta"]["guest_cohort"] = legacy_meta["cohort"]
        if "max_unique" in legacy_meta:
            state["_meta"]["guest_max_unique"] = legacy_meta["max_unique"]
        if "cohort_seed" in legacy_meta:
            state["_meta"]["cohort_seed"] = legacy_meta["cohort_seed"]
    return state


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_state()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return default_state()
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("State file must contain a JSON object.")
    return migrate_state(raw)


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def ensure_state_shape(state: dict[str, Any]) -> None:
    migrated = migrate_state(state)
    state.clear()
    state.update(migrated)


def get_role_bucket(state: dict[str, Any], role: str) -> dict[str, dict[str, int]]:
    if role not in ROLE_NAMES:
        raise ValueError(f"Unknown role '{role}'. Expected one of: {ROLE_NAMES}")
    ensure_state_shape(state)
    return state[role]


def ensure_role_records(state: dict[str, Any], role: str, names: list[str]) -> None:
    bucket = get_role_bucket(state, role)
    expected = set(names)

    for name in names:
        if name not in bucket:
            bucket[name] = RoleStats().to_mapping()
        else:
            bucket[name] = RoleStats.from_mapping(bucket[name]).to_mapping()

    for stale_name in list(bucket.keys()):
        if stale_name not in expected:
            del bucket[stale_name]


def score_key(role_bucket: dict[str, dict[str, int]], name: str) -> tuple[int, int]:
    stats = RoleStats.from_mapping(role_bucket.get(name))
    return (stats.assigned_count, stats.no_show_count)


def advance_role_cooldowns(state: dict[str, Any], role: str) -> None:
    bucket = get_role_bucket(state, role)
    for name in bucket:
        stats = RoleStats.from_mapping(bucket[name])
        if stats.cooldown > 0:
            stats.cooldown -= 1
        bucket[name] = stats.to_mapping()


def apply_role_outcomes(
    state: dict[str, Any],
    role: str,
    attended: list[str],
    no_show: list[str],
) -> None:
    bucket = get_role_bucket(state, role)
    attended_set = set(attended)
    no_show_set = set(no_show)
    overlap = attended_set & no_show_set
    if overlap:
        raise ValueError(
            f"Names listed in both attended and no_show for role '{role}': {sorted(overlap)[:10]}"
        )

    unknown = sorted((attended_set | no_show_set) - set(bucket.keys()))
    if unknown:
        raise ValueError(f"Unknown names in {role} outcome files: {unknown[:10]}")

    for name in attended_set:
        stats = RoleStats.from_mapping(bucket[name])
        stats.assigned_count += 1
        bucket[name] = stats.to_mapping()

    for name in no_show_set:
        stats = RoleStats.from_mapping(bucket[name])
        stats.assigned_count += 1
        stats.no_show_count += 1
        stats.cooldown = max(stats.cooldown, 1)
        bucket[name] = stats.to_mapping()
