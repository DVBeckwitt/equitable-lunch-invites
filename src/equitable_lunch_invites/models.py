from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DISCIPLINES = [
    "biophysics",
    "astronomy",
    "condensed matter experimental",
    "condensed matter theory",
]

DISCIPLINE_ALIASES = {
    "condensed matter exp": "condensed matter experimental",
    "cm experimental": "condensed matter experimental",
    "condensed matter experimental": "condensed matter experimental",
    "condensed matter theory": "condensed matter theory",
    "cm theory": "condensed matter theory",
    "biophys": "biophysics",
    "astro": "astronomy",
}

DEFAULT_GUEST_DEMOGRAPHIC_COLUMN = "Sex"
DEFAULT_GUEST_OUTCOME_COLUMN = "Outcome"
DEFAULT_WAITLIST_SIZE = 8
DEMOGRAPHIC_MODE_PROPORTIONAL = "proportional"
DEMOGRAPHIC_MODE_WOMEN_TO_PARITY = "women_to_parity"
DEFAULT_DEMOGRAPHIC_MODE = DEMOGRAPHIC_MODE_WOMEN_TO_PARITY
STATE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Participant:
    name: str
    discipline: str
    demographic: str | None = None


@dataclass
class RoleStats:
    assigned_count: int = 0
    no_show_count: int = 0
    cooldown: int = 0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RoleStats":
        raw = raw or {}
        return cls(
            assigned_count=int(raw.get("assigned_count", raw.get("attended_count", 0)) or 0),
            no_show_count=int(raw.get("no_show_count", 0) or 0),
            cooldown=int(raw.get("cooldown", 0) or 0),
        )

    def to_mapping(self) -> dict[str, int]:
        return {
            "assigned_count": int(self.assigned_count),
            "no_show_count": int(self.no_show_count),
            "cooldown": int(self.cooldown),
        }


@dataclass(frozen=True)
class PlannerConfig:
    event_index: int
    total_events: int
    hosts_per_event: int
    guests_per_event: int
    guest_max_unique: int
    seed: int
    cohort_seed: int
    guest_demographic_column: str = DEFAULT_GUEST_DEMOGRAPHIC_COLUMN
    demographic_mode: str = DEFAULT_DEMOGRAPHIC_MODE
    waitlist_size: int = DEFAULT_WAITLIST_SIZE


@dataclass(frozen=True)
class EventPlan:
    selected_hosts: list[Participant]
    waitlist_hosts: list[Participant]
    selected_guests: list[Participant]
    waitlist_guests: list[Participant]
    info: dict[str, Any]
