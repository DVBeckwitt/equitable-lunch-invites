from __future__ import annotations

import csv
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from equitable_lunch_invites.cli import _derive_seed
from equitable_lunch_invites.models import (
    DEMOGRAPHIC_MODE_PROPORTIONAL,
    DEMOGRAPHIC_MODE_WOMEN_TO_PARITY,
    DISCIPLINES,
    Participant,
    PlannerConfig,
)
from equitable_lunch_invites.selection import allocate_counts_deterministic, apply_demographic_mode, plan_event
from equitable_lunch_invites.state import (
    ROLE_GUESTS,
    ROLE_HOSTS,
    advance_role_cooldowns,
    apply_role_outcomes,
    default_state,
    ensure_role_records,
    ensure_state_shape,
)

METRIC_PANEL_DEFS: list[tuple[str, str]] = [
    ("female_share", "Female share"),
    ("female_to_male_invite_rate_ratio", "F/M invite rate ratio"),
    ("discipline_total_variation", "Discipline TV distance"),
    ("coverage_fraction", "Coverage fraction"),
    ("assignment_gini_gap", "Assignment gini gap"),
]


def _discipline_code(discipline: str) -> str:
    return "".join(part[:1] for part in (discipline or "").split()).lower() or "x"


def _make_host_roster() -> list[Participant]:
    hosts: list[Participant] = []
    for discipline in DISCIPLINES:
        code = _discipline_code(discipline)
        for idx in range(2):
            hosts.append(Participant(name=f"host-{code}-{idx:02d}", discipline=discipline))
    return hosts


def _make_guest_roster(
    scenario_id: str,
    discipline_sizes: dict[str, int],
    *,
    women_count: int = 12,
    men_count: int = 60,
    roster_seed: int = 0,
) -> list[Participant]:
    rng = random.Random(int(roster_seed))
    sizes = {discipline: int(discipline_sizes.get(discipline, 0) or 0) for discipline in DISCIPLINES}
    total = sum(sizes.values())
    if int(women_count) + int(men_count) != total:
        raise ValueError(
            f"Scenario '{scenario_id}' discipline sizes sum to {total}, "
            f"but requested women+men is {women_count + men_count}."
        )

    active_disciplines = [discipline for discipline in DISCIPLINES if sizes[discipline] > 0]
    women_by_discipline = {discipline: 0 for discipline in DISCIPLINES}
    if active_disciplines and women_count >= len(active_disciplines) and men_count >= len(active_disciplines):
        for discipline in active_disciplines:
            if sizes[discipline] >= 2:
                women_by_discipline[discipline] = 1
        remaining_women = int(women_count) - sum(women_by_discipline.values())
    else:
        remaining_women = int(women_count)

    while remaining_women > 0:
        candidates: list[tuple[int, str]] = []
        for discipline in active_disciplines:
            # Cap women so each discipline keeps at least one man (when possible).
            size = sizes[discipline]
            cap = max(0, size - 1) if size >= 2 else size
            capacity = cap - women_by_discipline[discipline]
            if capacity > 0:
                candidates.append((capacity, discipline))
        if not candidates:
            break

        # Weight by remaining capacity so larger disciplines are more likely to receive extra women.
        total_capacity = sum(capacity for capacity, _ in candidates)
        pick = rng.randrange(total_capacity)
        chosen = candidates[-1][1]
        for capacity, discipline in candidates:
            if pick < capacity:
                chosen = discipline
                break
            pick -= capacity
        women_by_discipline[chosen] += 1
        remaining_women -= 1

    if sum(women_by_discipline.values()) != int(women_count):
        raise ValueError(
            f"Could not allocate {women_count} women across disciplines for scenario '{scenario_id}'."
        )

    guests: list[Participant] = []
    for discipline in DISCIPLINES:
        size = sizes[discipline]
        if size <= 0:
            continue
        code = _discipline_code(discipline)
        women_in_discipline = int(women_by_discipline.get(discipline, 0) or 0)
        women_in_discipline = max(0, min(size, women_in_discipline))

        indices = list(range(size))
        rng.shuffle(indices)
        women_indices = set(indices[:women_in_discipline])
        for idx in range(size):
            guests.append(
                Participant(
                    name=f"{scenario_id}-guest-{code}-{idx:03d}",
                    discipline=discipline,
                    demographic=("F" if idx in women_indices else "M"),
                )
            )
    return guests


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in (value or ""))
    return "_".join(part for part in cleaned.split("_") if part) or "x"


def _gini(values: list[int]) -> float:
    cleaned = [max(0, int(value)) for value in values]
    if not cleaned:
        return 0.0
    total = sum(cleaned)
    if total <= 0:
        return 0.0

    cleaned.sort()
    n = len(cleaned)
    weighted_sum = 0
    for idx, value in enumerate(cleaned, start=1):
        weighted_sum += idx * value

    return (2.0 * weighted_sum) / (n * total) - (n + 1) / n


def _ideal_assignment_counts(total_seats: int, n_people: int) -> list[int]:
    if n_people <= 0:
        return []
    total = max(0, int(total_seats))
    q, r = divmod(total, int(n_people))
    return [q + 1] * r + [q] * (int(n_people) - r)


def _expected_demographic_totals(
    guest_roster: list[Participant],
    demographic_mode: str,
    total_seats: int,
) -> dict[str, int]:
    if total_seats <= 0:
        return {}

    demographic_counts = Counter(participant.demographic or "U" for participant in guest_roster)
    if demographic_mode == DEMOGRAPHIC_MODE_WOMEN_TO_PARITY:
        weights = apply_demographic_mode(
            demographic_counts={key: int(value) for key, value in demographic_counts.items()},
            demographic_column="Sex",
            demographic_mode=demographic_mode,
        )
    else:
        weights = {key: int(value) for key, value in demographic_counts.items()}

    return allocate_counts_deterministic(total_seats, weights)


def _expected_female_share(
    guest_roster: list[Participant],
    demographic_mode: str,
    total_seats: int,
) -> float:
    expected_totals = _expected_demographic_totals(
        guest_roster=guest_roster,
        demographic_mode=demographic_mode,
        total_seats=total_seats,
    )
    return float(expected_totals.get("F", 0)) / float(total_seats or 1)


@dataclass(frozen=True)
class TrialResult:
    scenario: str
    demographic_mode: str
    seed_key: int
    female_count: int
    male_count: int
    female_seats: int
    male_seats: int
    female_share: float
    expected_female_share: float
    roster_female_share: float
    female_share_minus_roster_share: float
    female_share_minus_expected_share: float
    female_invite_rate: float
    male_invite_rate: float
    female_to_male_invite_rate_ratio: float
    expected_female_to_male_invite_rate_ratio: float
    mean_abs_female_gap_per_event: float
    max_abs_discipline_share_diff: float
    discipline_total_variation: float
    discipline_coverage_min: float
    unique_guest_fraction: float
    coverage_fraction: float
    assignment_gini_gap: float
    assignment_gini_gap_female: float
    assignment_gini_gap_male: float


@dataclass(frozen=True)
class DisciplineResult:
    scenario: str
    demographic_mode: str
    seed_key: int
    discipline: str
    roster_count: int
    roster_female_count: int
    roster_male_count: int
    roster_female_share: float
    roster_share: float
    selected_count: int
    selected_share: float
    share_ratio: float
    invite_rate: float
    event_coverage: float


def _run_trial(
    *,
    scenario: str = "",
    host_roster: list[Participant],
    guest_roster: list[Participant],
    demographic_mode: str,
    seed_key: int,
    total_events: int,
    hosts_per_event: int,
    guests_per_event: int,
) -> tuple[TrialResult, list[DisciplineResult]]:
    state = default_state()
    ensure_state_shape(state)
    ensure_role_records(state, ROLE_HOSTS, [participant.name for participant in host_roster])
    ensure_role_records(state, ROLE_GUESTS, [participant.name for participant in guest_roster])
    state["_meta"]["seed_key"] = int(seed_key)

    total_seats = total_events * guests_per_event
    expected_totals = _expected_demographic_totals(
        guest_roster=guest_roster,
        demographic_mode=demographic_mode,
        total_seats=total_seats,
    )
    expected_female_seats = int(expected_totals.get("F", 0) or 0)
    expected_male_seats = int(expected_totals.get("M", 0) or 0)
    expected_f_share = float(expected_female_seats) / float(total_seats or 1)

    demographic_counts = Counter(participant.demographic or "U" for participant in guest_roster)
    female_count = int(demographic_counts.get("F", 0) or 0)
    male_count = int(demographic_counts.get("M", 0) or 0)
    roster_f_share = float(female_count) / float(len(guest_roster) or 1)

    total_female = 0
    abs_gap_sum = 0.0
    selected_demographics: Counter[str] = Counter()
    selected_disciplines = Counter()
    discipline_event_presence: Counter[str] = Counter()
    selected_names: set[str] = set()

    cohort_seed = _derive_seed(seed_key, "cohort")
    for event_index in range(1, total_events + 1):
        config = PlannerConfig(
            event_index=event_index,
            total_events=total_events,
            hosts_per_event=hosts_per_event,
            guests_per_event=guests_per_event,
            guest_max_unique=len(guest_roster),
            seed=_derive_seed(seed_key, f"event:{event_index}"),
            cohort_seed=cohort_seed,
            guest_demographic_column="Sex",
            demographic_mode=demographic_mode,
            waitlist_size=0,
        )
        event_plan = plan_event(
            host_roster=host_roster,
            guest_roster=guest_roster,
            state=state,
            config=config,
        )

        selected_guests = event_plan.selected_guests
        assert len(selected_guests) == guests_per_event

        event_female = sum(1 for participant in selected_guests if (participant.demographic or "U") == "F")
        total_female += event_female

        targets = event_plan.info.get("guest_demographic_targets", {}) or {}
        target_female = int(targets.get("F", 0) or 0)
        abs_gap_sum += abs(event_female - target_female)

        selected_demographics.update(participant.demographic or "U" for participant in selected_guests)
        selected_disciplines.update(participant.discipline for participant in selected_guests)
        selected_names.update(participant.name for participant in selected_guests)
        for discipline in {participant.discipline for participant in selected_guests}:
            discipline_event_presence[discipline] += 1

        advance_role_cooldowns(state, ROLE_GUESTS)
        apply_role_outcomes(
            state,
            ROLE_HOSTS,
            attended=[participant.name for participant in event_plan.selected_hosts],
            no_show=[],
        )
        apply_role_outcomes(
            state,
            ROLE_GUESTS,
            attended=[participant.name for participant in selected_guests],
            no_show=[],
            cant_attend=[],
        )
        state["_meta"]["last_recorded_event"] = event_index

    female_share = float(total_female) / float(total_seats or 1)
    female_seats = int(selected_demographics.get("F", 0) or 0)
    male_seats = int(selected_demographics.get("M", 0) or 0)
    mean_abs_gap = float(abs_gap_sum) / float(total_events or 1)

    roster_discipline_counts = Counter(participant.discipline for participant in guest_roster)
    max_abs_discipline_diff = 0.0
    discipline_l1 = 0.0
    discipline_coverage_min = 1.0
    for discipline in roster_discipline_counts:
        roster_share = float(roster_discipline_counts[discipline]) / float(len(guest_roster) or 1)
        selected_share = float(selected_disciplines.get(discipline, 0)) / float(total_seats or 1)
        diff = abs(selected_share - roster_share)
        max_abs_discipline_diff = max(max_abs_discipline_diff, diff)
        discipline_l1 += diff
        coverage = float(discipline_event_presence.get(discipline, 0)) / float(total_events or 1)
        discipline_coverage_min = min(discipline_coverage_min, coverage)

    discipline_total_variation = 0.5 * discipline_l1

    unique_guest_fraction = float(len(selected_names)) / float(total_seats or 1)
    coverage_fraction = float(len(selected_names)) / float(len(guest_roster) or 1)

    assigned_counts_by_name = {
        name: int(stats.get("assigned_count", 0) or 0)
        for name, stats in state.get(ROLE_GUESTS, {}).items()
        if isinstance(stats, dict)
    }
    assignment_counts = [assigned_counts_by_name.get(participant.name, 0) for participant in guest_roster]
    assignment_gini = _gini(assignment_counts)
    assignment_gini_ideal = _gini(_ideal_assignment_counts(total_seats, len(guest_roster)))
    assignment_gini_gap = max(0.0, assignment_gini - assignment_gini_ideal)

    female_names = [participant.name for participant in guest_roster if (participant.demographic or "U") == "F"]
    male_names = [participant.name for participant in guest_roster if (participant.demographic or "U") == "M"]
    female_assignment_counts = [assigned_counts_by_name.get(name, 0) for name in female_names]
    male_assignment_counts = [assigned_counts_by_name.get(name, 0) for name in male_names]
    assignment_gini_female = _gini(female_assignment_counts)
    assignment_gini_male = _gini(male_assignment_counts)
    assignment_gini_ideal_female = _gini(_ideal_assignment_counts(female_seats, len(female_names)))
    assignment_gini_ideal_male = _gini(_ideal_assignment_counts(male_seats, len(male_names)))
    assignment_gini_gap_female = max(0.0, assignment_gini_female - assignment_gini_ideal_female)
    assignment_gini_gap_male = max(0.0, assignment_gini_male - assignment_gini_ideal_male)

    female_share_minus_roster = female_share - roster_f_share
    female_share_minus_expected = female_share - expected_f_share

    female_invite_rate = float(female_seats) / float(female_count or 1)
    male_invite_rate = float(male_seats) / float(male_count or 1)
    female_to_male_invite_rate_ratio = (
        (female_invite_rate / male_invite_rate) if male_invite_rate > 0 else float("inf")
    )

    expected_female_invite_rate = float(expected_female_seats) / float(female_count or 1)
    expected_male_invite_rate = float(expected_male_seats) / float(male_count or 1)
    expected_invite_rate_ratio = (
        (expected_female_invite_rate / expected_male_invite_rate) if expected_male_invite_rate > 0 else float("inf")
    )

    discipline_rows: list[DisciplineResult] = []
    roster_female_by_discipline = Counter(
        participant.discipline for participant in guest_roster if (participant.demographic or "U") == "F"
    )
    roster_male_by_discipline = Counter(
        participant.discipline for participant in guest_roster if (participant.demographic or "U") == "M"
    )
    for discipline in roster_discipline_counts:
        roster_count = int(roster_discipline_counts.get(discipline, 0) or 0)
        roster_female_count = int(roster_female_by_discipline.get(discipline, 0) or 0)
        roster_male_count = int(roster_male_by_discipline.get(discipline, 0) or 0)
        roster_female_share = float(roster_female_count) / float(roster_count or 1)
        roster_share = float(roster_count) / float(len(guest_roster) or 1)
        selected_count = int(selected_disciplines.get(discipline, 0) or 0)
        selected_share = float(selected_count) / float(total_seats or 1)
        share_ratio = (selected_share / roster_share) if roster_share > 0 else 0.0
        invite_rate = float(selected_count) / float(roster_count or 1)
        event_coverage = float(discipline_event_presence.get(discipline, 0) or 0) / float(total_events or 1)
        discipline_rows.append(
            DisciplineResult(
                scenario=scenario,
                demographic_mode=demographic_mode,
                seed_key=seed_key,
                discipline=discipline,
                roster_count=roster_count,
                roster_female_count=roster_female_count,
                roster_male_count=roster_male_count,
                roster_female_share=roster_female_share,
                roster_share=roster_share,
                selected_count=selected_count,
                selected_share=selected_share,
                share_ratio=share_ratio,
                invite_rate=invite_rate,
                event_coverage=event_coverage,
            )
        )

    result = TrialResult(
        scenario=scenario,
        demographic_mode=demographic_mode,
        seed_key=seed_key,
        female_count=female_count,
        male_count=male_count,
        female_seats=female_seats,
        male_seats=male_seats,
        female_share=female_share,
        expected_female_share=expected_f_share,
        roster_female_share=roster_f_share,
        female_share_minus_roster_share=female_share_minus_roster,
        female_share_minus_expected_share=female_share_minus_expected,
        female_invite_rate=female_invite_rate,
        male_invite_rate=male_invite_rate,
        female_to_male_invite_rate_ratio=female_to_male_invite_rate_ratio,
        expected_female_to_male_invite_rate_ratio=expected_invite_rate_ratio,
        mean_abs_female_gap_per_event=mean_abs_gap,
        max_abs_discipline_share_diff=max_abs_discipline_diff,
        discipline_total_variation=discipline_total_variation,
        discipline_coverage_min=discipline_coverage_min,
        unique_guest_fraction=unique_guest_fraction,
        coverage_fraction=coverage_fraction,
        assignment_gini_gap=assignment_gini_gap,
        assignment_gini_gap_female=assignment_gini_gap_female,
        assignment_gini_gap_male=assignment_gini_gap_male,
    )
    return result, discipline_rows


def _write_csv(path: Path, rows: list[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "demographic_mode",
                "seed_key",
                "female_count",
                "male_count",
                "female_seats",
                "male_seats",
                "female_share",
                "roster_female_share",
                "expected_female_share",
                "female_share_minus_roster_share",
                "female_share_minus_expected_share",
                "female_invite_rate",
                "male_invite_rate",
                "female_to_male_invite_rate_ratio",
                "expected_female_to_male_invite_rate_ratio",
                "coverage_fraction",
                "mean_abs_female_gap_per_event",
                "max_abs_discipline_share_diff",
                "discipline_total_variation",
                "discipline_coverage_min",
                "unique_guest_fraction",
                "assignment_gini_gap",
                "assignment_gini_gap_female",
                "assignment_gini_gap_male",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scenario": row.scenario,
                    "demographic_mode": row.demographic_mode,
                    "seed_key": row.seed_key,
                    "female_count": row.female_count,
                    "male_count": row.male_count,
                    "female_seats": row.female_seats,
                    "male_seats": row.male_seats,
                    "female_share": f"{row.female_share:.6f}",
                    "roster_female_share": f"{row.roster_female_share:.6f}",
                    "expected_female_share": f"{row.expected_female_share:.6f}",
                    "female_share_minus_roster_share": f"{row.female_share_minus_roster_share:.6f}",
                    "female_share_minus_expected_share": f"{row.female_share_minus_expected_share:.6f}",
                    "female_invite_rate": f"{row.female_invite_rate:.6f}",
                    "male_invite_rate": f"{row.male_invite_rate:.6f}",
                    "female_to_male_invite_rate_ratio": f"{row.female_to_male_invite_rate_ratio:.6f}",
                    "expected_female_to_male_invite_rate_ratio": f"{row.expected_female_to_male_invite_rate_ratio:.6f}",
                    "coverage_fraction": f"{row.coverage_fraction:.6f}",
                    "mean_abs_female_gap_per_event": f"{row.mean_abs_female_gap_per_event:.6f}",
                    "max_abs_discipline_share_diff": f"{row.max_abs_discipline_share_diff:.6f}",
                    "discipline_total_variation": f"{row.discipline_total_variation:.6f}",
                    "discipline_coverage_min": f"{row.discipline_coverage_min:.6f}",
                    "unique_guest_fraction": f"{row.unique_guest_fraction:.6f}",
                    "assignment_gini_gap": f"{row.assignment_gini_gap:.6f}",
                    "assignment_gini_gap_female": f"{row.assignment_gini_gap_female:.6f}",
                    "assignment_gini_gap_male": f"{row.assignment_gini_gap_male:.6f}",
                }
            )


def _write_discipline_csv(path: Path, rows: list[DisciplineResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "demographic_mode",
                "seed_key",
                "discipline",
                "roster_count",
                "roster_female_count",
                "roster_male_count",
                "roster_female_share",
                "roster_share",
                "selected_count",
                "selected_share",
                "share_ratio",
                "invite_rate",
                "event_coverage",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scenario": row.scenario,
                    "demographic_mode": row.demographic_mode,
                    "seed_key": row.seed_key,
                    "discipline": row.discipline,
                    "roster_count": row.roster_count,
                    "roster_female_count": row.roster_female_count,
                    "roster_male_count": row.roster_male_count,
                    "roster_female_share": f"{row.roster_female_share:.6f}",
                    "roster_share": f"{row.roster_share:.6f}",
                    "selected_count": row.selected_count,
                    "selected_share": f"{row.selected_share:.6f}",
                    "share_ratio": f"{row.share_ratio:.6f}",
                    "invite_rate": f"{row.invite_rate:.6f}",
                    "event_coverage": f"{row.event_coverage:.6f}",
                }
            )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo]) * (1.0 - weight) + float(ordered[hi]) * weight


def _write_summary_csv(path: Path, rows: list[TrialResult]) -> None:
    grouped: dict[tuple[str, str], list[TrialResult]] = defaultdict(list)
    for row in rows:
        grouped[(row.scenario, row.demographic_mode)].append(row)

    metrics = [
        ("female_share", lambda row: row.female_share),
        ("female_to_male_invite_rate_ratio", lambda row: row.female_to_male_invite_rate_ratio),
        ("discipline_total_variation", lambda row: row.discipline_total_variation),
        ("coverage_fraction", lambda row: row.coverage_fraction),
        ("assignment_gini_gap", lambda row: row.assignment_gini_gap),
        ("assignment_gini_gap_female", lambda row: row.assignment_gini_gap_female),
        ("assignment_gini_gap_male", lambda row: row.assignment_gini_gap_male),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "demographic_mode",
                "trials",
                *[f"{name}_mean" for name, _ in metrics],
                *[f"{name}_p10" for name, _ in metrics],
                *[f"{name}_p50" for name, _ in metrics],
                *[f"{name}_p90" for name, _ in metrics],
            ],
        )
        writer.writeheader()
        for (scenario, mode), bucket in sorted(grouped.items()):
            row_out: dict[str, object] = {
                "scenario": scenario,
                "demographic_mode": mode,
                "trials": len(bucket),
            }
            for name, getter in metrics:
                values = [float(getter(row)) for row in bucket]
                row_out[f"{name}_mean"] = f"{mean(values):.6f}"
                row_out[f"{name}_p10"] = f"{_quantile(values, 0.10):.6f}"
                row_out[f"{name}_p50"] = f"{_quantile(values, 0.50):.6f}"
                row_out[f"{name}_p90"] = f"{_quantile(values, 0.90):.6f}"
            writer.writerow(row_out)


def _write_report(
    path: Path,
    rows: list[TrialResult],
    discipline_rows: list[DisciplineResult],
) -> None:
    by_scenario_mode: dict[tuple[str, str], list[TrialResult]] = defaultdict(list)
    for row in rows:
        by_scenario_mode[(row.scenario, row.demographic_mode)].append(row)

    roster_by_scenario: dict[str, dict[str, DisciplineResult]] = defaultdict(dict)
    for row in discipline_rows:
        roster_by_scenario[row.scenario].setdefault(row.discipline, row)

    def fmt(values: list[float]) -> str:
        return (
            f"{mean(values):.3f} "
            f"({ _quantile(values, 0.10):.3f}–{ _quantile(values, 0.90):.3f})"
        )

    scenarios = sorted({row.scenario for row in rows})
    modes = [DEMOGRAPHIC_MODE_PROPORTIONAL, DEMOGRAPHIC_MODE_WOMEN_TO_PARITY]

    lines: list[str] = []
    lines.append("# Representation trial report")
    lines.append("")
    lines.append("These simulations use mock guest rosters with **12 women** (`F`) and **60 men** (`M`).")
    lines.append(
        "Women/men are randomly distributed across disciplines (each discipline includes at least one woman and one man)."
    )
    lines.append("")
    lines.append("Generated figures:")
    lines.append("- `representation_trials.png`: key metrics by scenario (boxplots).")
    lines.append("- `representation_disciplines.png`: discipline seat-share ratio by scenario/discipline (boxplots).")
    lines.append("- `representation_metric_<metric>__<mode>.png`: one file per metric panel.")
    lines.append(
        "- `representation_discipline_share_ratio__<scenario>__<mode>.png`: one file per discipline panel."
    )
    lines.append("")
    lines.append("## Metrics (what they mean)")
    lines.append("- `female_share`: fraction of selected guest seats filled by women (`F`).")
    lines.append(
        "- `female_to_male_invite_rate_ratio`: per-person invite rate ratio = (F seats/F people) ÷ (M seats/M people)."
    )
    lines.append(
        "  - `1.0` means equal invites per person; `5.0` means women are invited 5× as often per person as men."
    )
    lines.append(
        "- `discipline_total_variation`: how far selected discipline shares drift from roster discipline shares (0 is perfect match)."
    )
    lines.append("- `coverage_fraction`: unique guests selected at least once ÷ roster size (1 means everyone got invited at least once).")
    lines.append(
        "- `assignment_gini_gap`: assignment Gini - ideal Gini (0 is as even a rotation as possible given seats/roster)."
    )
    lines.append("- `assignment_gini_gap_female`/`assignment_gini_gap_male`: same within women/men groups.")
    lines.append("")
    lines.append("## What to look for (plain English)")
    lines.append(
        "- In `proportional`, the target is to match the roster: women get ~16.7% of seats and per-capita invites are ~equal (ratio ~1.0)."
    )
    lines.append(
        "- In `women_to_parity`, the target is 50/50 seats even with fewer women: per-capita invites for women should be much higher (ratio ~5.0 for 12F/60M)."
    )
    lines.append(
        "- If `women_to_parity` hits its target, it improves representation, but it can reduce coverage (more repeats) and increase unevenness because the women pool is smaller."
    )
    lines.append(
        "- Discipline plots/metrics can drift from roster shares because the planner anchors disciplines to avoid drop-out (small disciplines can be overrepresented)."
    )
    lines.append("")

    for scenario in scenarios:
        lines.append(f"## Scenario: {scenario}")
        lines.append("")
        roster_disciplines = roster_by_scenario.get(scenario, {})
        if roster_disciplines:
            lines.append("Roster by discipline (randomized sex placement):")
            lines.append("")
            lines.append("| discipline | total | women | men | women% |")
            lines.append("|---|---:|---:|---:|---:|")
            for discipline in DISCIPLINES:
                row = roster_disciplines.get(discipline)
                if not row:
                    continue
                lines.append(
                    f"| {discipline} | {row.roster_count} | {row.roster_female_count} | "
                    f"{row.roster_male_count} | {row.roster_female_share * 100.0:.1f}% |"
                )
            lines.append("")

        mode_means: dict[str, dict[str, float]] = {}
        for mode in modes:
            bucket = by_scenario_mode.get((scenario, mode), [])
            if not bucket:
                continue

            expected_share = float(bucket[0].expected_female_share)
            expected_ratio = float(bucket[0].expected_female_to_male_invite_rate_ratio)
            roster_share = float(bucket[0].roster_female_share)

            lines.append(f"### {mode}")
            lines.append("")
            lines.append(f"Targets: female_share={expected_share:.3f}, F/M invite rate ratio={expected_ratio:.3f}.")
            lines.append(f"Roster: female_share={roster_share:.3f} (12/72).")
            lines.append("")
            lines.append(f"- female_share: {fmt([row.female_share for row in bucket])}")
            lines.append(f"- F/M invite rate ratio: {fmt([row.female_to_male_invite_rate_ratio for row in bucket])}")
            lines.append(f"- discipline_total_variation: {fmt([row.discipline_total_variation for row in bucket])}")
            lines.append(f"- coverage_fraction: {fmt([row.coverage_fraction for row in bucket])}")
            lines.append(f"- assignment_gini_gap: {fmt([row.assignment_gini_gap for row in bucket])}")
            lines.append(f"- assignment_gini_gap_female: {fmt([row.assignment_gini_gap_female for row in bucket])}")
            lines.append(f"- assignment_gini_gap_male: {fmt([row.assignment_gini_gap_male for row in bucket])}")
            lines.append("")

            mode_means[mode] = {
                "female_share": float(mean([row.female_share for row in bucket])),
                "invite_ratio": float(mean([row.female_to_male_invite_rate_ratio for row in bucket])),
                "discipline_tv": float(mean([row.discipline_total_variation for row in bucket])),
                "coverage": float(mean([row.coverage_fraction for row in bucket])),
                "gini_gap": float(mean([row.assignment_gini_gap for row in bucket])),
            }

        if DEMOGRAPHIC_MODE_PROPORTIONAL in mode_means and DEMOGRAPHIC_MODE_WOMEN_TO_PARITY in mode_means:
            prop = mode_means[DEMOGRAPHIC_MODE_PROPORTIONAL]
            parity = mode_means[DEMOGRAPHIC_MODE_WOMEN_TO_PARITY]
            lines.append("### women_to_parity vs proportional (mean deltas)")
            lines.append("")
            lines.append(f"- female_share: {parity['female_share'] - prop['female_share']:+.3f}")
            lines.append(f"- F/M invite rate ratio: {parity['invite_ratio'] - prop['invite_ratio']:+.3f}")
            lines.append(f"- discipline_total_variation: {parity['discipline_tv'] - prop['discipline_tv']:+.3f}")
            lines.append(f"- coverage_fraction: {parity['coverage'] - prop['coverage']:+.3f}")
            lines.append(f"- assignment_gini_gap: {parity['gini_gap'] - prop['gini_gap']:+.3f}")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _try_write_plot(
    path: Path,
    scenarios: list[str],
    results: list[TrialResult],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    modes = [DEMOGRAPHIC_MODE_PROPORTIONAL, DEMOGRAPHIC_MODE_WOMEN_TO_PARITY]

    roster_share_by_scenario: dict[str, float] = {}
    expected_share_by_mode: dict[str, dict[str, float]] = defaultdict(dict)
    expected_ratio_by_mode: dict[str, dict[str, float]] = defaultdict(dict)
    for row in results:
        roster_share_by_scenario.setdefault(row.scenario, row.roster_female_share)
        expected_share_by_mode[row.demographic_mode].setdefault(row.scenario, row.expected_female_share)
        expected_ratio_by_mode[row.demographic_mode].setdefault(
            row.scenario, row.expected_female_to_male_invite_rate_ratio
        )

    metric_rows = [
        ("Female share", "female_share"),
        ("F/M invite rate ratio", "female_to_male_invite_rate_ratio"),
        ("Discipline TV distance", "discipline_total_variation"),
        ("Coverage fraction", "coverage_fraction"),
        ("Assignment gini gap", "assignment_gini_gap"),
    ]

    fig, axes = plt.subplots(
        len(metric_rows),
        len(modes),
        figsize=(13, 12),
        sharex=True,
        constrained_layout=True,
    )
    for col_idx, mode in enumerate(modes):
        mode_bucket = [row for row in results if row.demographic_mode == mode]
        axes[0][col_idx].set_title(mode)

        for row_idx, (label, attr) in enumerate(metric_rows):
            ax = axes[row_idx][col_idx]
            box_data = [
                [float(getattr(row, attr)) for row in mode_bucket if row.scenario == scenario]
                for scenario in scenarios
            ]
            ax.boxplot(box_data, showmeans=True)
            ax.grid(axis="y", alpha=0.25)

            if attr == "female_share":
                ax.set_ylim(0.0, 1.0)
                ax.plot(
                    range(1, len(scenarios) + 1),
                    [roster_share_by_scenario.get(scenario, 0.0) for scenario in scenarios],
                    "k--",
                    linewidth=1.0,
                    label="roster",
                )
                ax.plot(
                    range(1, len(scenarios) + 1),
                    [expected_share_by_mode.get(mode, {}).get(scenario, 0.0) for scenario in scenarios],
                    "o",
                    label="target",
                )
                if col_idx == 0:
                    ax.legend(loc="lower right")
            elif attr == "female_to_male_invite_rate_ratio":
                ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
                ax.plot(
                    range(1, len(scenarios) + 1),
                    [expected_ratio_by_mode.get(mode, {}).get(scenario, 0.0) for scenario in scenarios],
                    "o",
                    label="target",
                )
                all_values = [value for series in box_data for value in series if math.isfinite(value)]
                all_values.extend(
                    value
                    for value in expected_ratio_by_mode.get(mode, {}).values()
                    if math.isfinite(float(value))
                )
                if all_values:
                    ax.set_ylim(0.0, max(2.0, max(all_values) * 1.1))
                if col_idx == 0 and row_idx == 1:
                    ax.legend(loc="lower right")
            elif attr == "discipline_total_variation":
                ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
                all_values = [value for series in box_data for value in series if math.isfinite(value)]
                if all_values:
                    ax.set_ylim(0.0, max(0.05, max(all_values) * 1.2))
            elif attr == "coverage_fraction":
                ax.set_ylim(0.0, 1.0)
                ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
            elif attr == "assignment_gini_gap":
                ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
                all_values = [value for series in box_data for value in series if math.isfinite(value)]
                if all_values:
                    ax.set_ylim(0.0, max(0.05, max(all_values) * 1.2))

            if col_idx == 0:
                ax.set_ylabel(label)
            if row_idx == len(metric_rows) - 1:
                ax.set_xticks(range(1, len(scenarios) + 1), scenarios, rotation=20, ha="right")
            else:
                ax.set_xticks(range(1, len(scenarios) + 1), [""] * len(scenarios))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _try_write_discipline_plot(
    path: Path,
    scenarios: list[str],
    discipline_rows: list[DisciplineResult],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    modes = [DEMOGRAPHIC_MODE_PROPORTIONAL, DEMOGRAPHIC_MODE_WOMEN_TO_PARITY]
    discipline_labels = [_slug(discipline) for discipline in DISCIPLINES]

    fig, axes = plt.subplots(
        len(scenarios),
        len(modes),
        figsize=(13, 9),
        sharey=True,
        constrained_layout=True,
    )

    for row_idx, scenario in enumerate(scenarios):
        for col_idx, mode in enumerate(modes):
            ax = axes[row_idx][col_idx]
            bucket = [
                row
                for row in discipline_rows
                if row.scenario == scenario and row.demographic_mode == mode
            ]
            by_discipline: dict[str, list[float]] = defaultdict(list)
            for row in bucket:
                by_discipline[row.discipline].append(float(row.share_ratio))

            box_data = [by_discipline.get(discipline, []) for discipline in DISCIPLINES]
            ax.boxplot(box_data, showmeans=True)
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
            ax.set_title(f"{scenario} / {mode}")
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(range(1, len(DISCIPLINES) + 1), discipline_labels, rotation=20, ha="right")
            if col_idx == 0:
                ax.set_ylabel("Seat share / roster share")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _metric_panel_filename(metric_attr: str, demographic_mode: str) -> str:
    return f"representation_metric_{_slug(metric_attr)}__{_slug(demographic_mode)}.png"


def _discipline_panel_filename(scenario: str, demographic_mode: str) -> str:
    return f"representation_discipline_share_ratio__{_slug(scenario)}__{_slug(demographic_mode)}.png"


def _try_write_metric_panel_plots(
    path_dir: Path,
    scenarios: list[str],
    results: list[TrialResult],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    path_dir.mkdir(parents=True, exist_ok=True)
    modes = [DEMOGRAPHIC_MODE_PROPORTIONAL, DEMOGRAPHIC_MODE_WOMEN_TO_PARITY]

    roster_share_by_scenario: dict[str, float] = {}
    expected_share_by_mode: dict[str, dict[str, float]] = defaultdict(dict)
    expected_ratio_by_mode: dict[str, dict[str, float]] = defaultdict(dict)
    for row in results:
        roster_share_by_scenario.setdefault(row.scenario, row.roster_female_share)
        expected_share_by_mode[row.demographic_mode].setdefault(row.scenario, row.expected_female_share)
        expected_ratio_by_mode[row.demographic_mode].setdefault(
            row.scenario, row.expected_female_to_male_invite_rate_ratio
        )

    for mode in modes:
        mode_bucket = [row for row in results if row.demographic_mode == mode]
        for metric_attr, metric_label in METRIC_PANEL_DEFS:
            box_data = [
                [float(getattr(row, metric_attr)) for row in mode_bucket if row.scenario == scenario]
                for scenario in scenarios
            ]

            fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
            ax.boxplot(box_data, showmeans=True)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(f"{metric_label} / {mode}")
            ax.set_ylabel(metric_label)
            ax.set_xticks(range(1, len(scenarios) + 1), scenarios, rotation=20, ha="right")

            if metric_attr == "female_share":
                ax.set_ylim(0.0, 1.0)
                ax.plot(
                    range(1, len(scenarios) + 1),
                    [roster_share_by_scenario.get(scenario, 0.0) for scenario in scenarios],
                    "k--",
                    linewidth=1.0,
                    label="roster",
                )
                ax.plot(
                    range(1, len(scenarios) + 1),
                    [expected_share_by_mode.get(mode, {}).get(scenario, 0.0) for scenario in scenarios],
                    "o",
                    label="target",
                )
                ax.legend(loc="lower right")
            elif metric_attr == "female_to_male_invite_rate_ratio":
                ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
                ax.plot(
                    range(1, len(scenarios) + 1),
                    [expected_ratio_by_mode.get(mode, {}).get(scenario, 0.0) for scenario in scenarios],
                    "o",
                    label="target",
                )
                all_values = [value for series in box_data for value in series if math.isfinite(value)]
                all_values.extend(
                    value
                    for value in expected_ratio_by_mode.get(mode, {}).values()
                    if math.isfinite(float(value))
                )
                if all_values:
                    ax.set_ylim(0.0, max(2.0, max(all_values) * 1.1))
                ax.legend(loc="lower right")
            elif metric_attr == "discipline_total_variation":
                ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
                all_values = [value for series in box_data for value in series if math.isfinite(value)]
                if all_values:
                    ax.set_ylim(0.0, max(0.05, max(all_values) * 1.2))
            elif metric_attr == "coverage_fraction":
                ax.set_ylim(0.0, 1.0)
                ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
            elif metric_attr == "assignment_gini_gap":
                ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
                all_values = [value for series in box_data for value in series if math.isfinite(value)]
                if all_values:
                    ax.set_ylim(0.0, max(0.05, max(all_values) * 1.2))

            panel_path = path_dir / _metric_panel_filename(metric_attr, mode)
            fig.savefig(panel_path, dpi=150)
            plt.close(fig)

    return True


def _try_write_discipline_panel_plots(
    path_dir: Path,
    scenarios: list[str],
    discipline_rows: list[DisciplineResult],
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    path_dir.mkdir(parents=True, exist_ok=True)
    modes = [DEMOGRAPHIC_MODE_PROPORTIONAL, DEMOGRAPHIC_MODE_WOMEN_TO_PARITY]
    discipline_labels = [_slug(discipline) for discipline in DISCIPLINES]

    for scenario in scenarios:
        for mode in modes:
            bucket = [
                row
                for row in discipline_rows
                if row.scenario == scenario and row.demographic_mode == mode
            ]
            by_discipline: dict[str, list[float]] = defaultdict(list)
            for row in bucket:
                by_discipline[row.discipline].append(float(row.share_ratio))

            box_data = [by_discipline.get(discipline, []) for discipline in DISCIPLINES]

            fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
            ax.boxplot(box_data, showmeans=True)
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
            ax.set_title(f"{scenario} / {mode}")
            ax.set_ylabel("Seat share / roster share")
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(range(1, len(DISCIPLINES) + 1), discipline_labels, rotation=20, ha="right")

            panel_path = path_dir / _discipline_panel_filename(scenario, mode)
            fig.savefig(panel_path, dpi=150)
            plt.close(fig)

    return True


def _mirror_figure_for_latex(path: Path, latex_figure_dir: Path) -> Path:
    latex_figure_dir.mkdir(parents=True, exist_ok=True)
    target = latex_figure_dir / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    return target


def test_representation_trials_run_and_plot() -> None:
    trials = int(os.environ.get("EQUITABLE_TRIALS", "40"))
    total_events = int(os.environ.get("EQUITABLE_TOTAL_EVENTS", "12"))
    guests_per_event = int(os.environ.get("EQUITABLE_GUESTS_PER_EVENT", "5"))
    hosts_per_event = int(os.environ.get("EQUITABLE_HOSTS_PER_EVENT", "1"))

    repo_root = Path(__file__).resolve().parents[1]
    default_artifact_dir = repo_root / "data" / "representation_artifacts"
    artifact_dir = Path(os.environ.get("EQUITABLE_ARTIFACT_DIR", str(default_artifact_dir)))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    latex_figure_dir_raw = str(os.environ.get("EQUITABLE_LATEX_FIGURE_DIR", "")).strip()
    latex_figure_dir = Path(latex_figure_dir_raw) if latex_figure_dir_raw else artifact_dir

    host_roster = _make_host_roster()
    scenarios = [
        (
            "even",
            _make_guest_roster(
                "even",
                {discipline: 18 for discipline in DISCIPLINES},
                roster_seed=_derive_seed(0, "roster:even"),
            ),
        ),
        (
            "discipline_imbalanced",
            _make_guest_roster(
                "discipline_imbalanced",
                {
                    DISCIPLINES[0]: 36,
                    DISCIPLINES[1]: 18,
                    DISCIPLINES[2]: 12,
                    DISCIPLINES[3]: 6,
                },
                roster_seed=_derive_seed(0, "roster:discipline_imbalanced"),
            ),
        ),
        (
            "discipline_extreme",
            _make_guest_roster(
                "discipline_extreme",
                {
                    DISCIPLINES[0]: 54,
                    DISCIPLINES[1]: 6,
                    DISCIPLINES[2]: 6,
                    DISCIPLINES[3]: 6,
                },
                roster_seed=_derive_seed(0, "roster:discipline_extreme"),
            ),
        ),
    ]
    for scenario_name, guest_roster in scenarios:
        sex_counts = Counter(participant.demographic or "U" for participant in guest_roster)
        assert sex_counts["F"] == 12, f"{scenario_name} should have 12 women, got {sex_counts['F']}"
        assert sex_counts["M"] == 60, f"{scenario_name} should have 60 men, got {sex_counts['M']}"
        for discipline in DISCIPLINES:
            bucket = [
                participant.demographic or "U"
                for participant in guest_roster
                if participant.discipline == discipline
            ]
            discipline_counts = Counter(bucket)
            assert discipline_counts["F"] > 0, f"{scenario_name} should include women in '{discipline}'"
            assert discipline_counts["M"] > 0, f"{scenario_name} should include men in '{discipline}'"

    all_results: list[TrialResult] = []
    all_discipline_results: list[DisciplineResult] = []
    for scenario_name, guest_roster in scenarios:
        for demographic_mode in (DEMOGRAPHIC_MODE_PROPORTIONAL, DEMOGRAPHIC_MODE_WOMEN_TO_PARITY):
            for seed_key in range(trials):
                result, discipline_rows = _run_trial(
                    scenario=scenario_name,
                    host_roster=host_roster,
                    guest_roster=guest_roster,
                    demographic_mode=demographic_mode,
                    seed_key=seed_key,
                    total_events=total_events,
                    hosts_per_event=hosts_per_event,
                    guests_per_event=guests_per_event,
                )
                all_results.append(result)
                all_discipline_results.extend(discipline_rows)

    csv_path = artifact_dir / "representation_trials.csv"
    _write_csv(csv_path, all_results)
    assert csv_path.exists()

    summary_csv_path = artifact_dir / "representation_trials_summary.csv"
    _write_summary_csv(summary_csv_path, all_results)
    assert summary_csv_path.exists()

    report_path = artifact_dir / "representation_trials_report.md"
    _write_report(report_path, all_results, all_discipline_results)
    assert report_path.exists()

    discipline_csv_path = artifact_dir / "representation_trials_disciplines.csv"
    _write_discipline_csv(discipline_csv_path, all_discipline_results)
    assert discipline_csv_path.exists()

    plot_path = artifact_dir / "representation_trials.png"
    wrote_plot = _try_write_plot(
        plot_path,
        scenarios=[name for name, _ in scenarios],
        results=all_results,
    )
    if wrote_plot:
        assert plot_path.exists()
        mirrored_plot_path = _mirror_figure_for_latex(plot_path, latex_figure_dir)
        assert mirrored_plot_path.exists()

    discipline_plot_path = artifact_dir / "representation_disciplines.png"
    wrote_discipline_plot = _try_write_discipline_plot(
        discipline_plot_path,
        scenarios=[name for name, _ in scenarios],
        discipline_rows=all_discipline_results,
    )
    if wrote_discipline_plot:
        assert discipline_plot_path.exists()
        mirrored_discipline_plot_path = _mirror_figure_for_latex(discipline_plot_path, latex_figure_dir)
        assert mirrored_discipline_plot_path.exists()

    wrote_metric_panels = _try_write_metric_panel_plots(
        artifact_dir,
        scenarios=[name for name, _ in scenarios],
        results=all_results,
    )
    if wrote_metric_panels:
        for metric_attr, _metric_label in METRIC_PANEL_DEFS:
            for mode in (DEMOGRAPHIC_MODE_PROPORTIONAL, DEMOGRAPHIC_MODE_WOMEN_TO_PARITY):
                assert (artifact_dir / _metric_panel_filename(metric_attr, mode)).exists()

    wrote_discipline_panels = _try_write_discipline_panel_plots(
        artifact_dir,
        scenarios=[name for name, _ in scenarios],
        discipline_rows=all_discipline_results,
    )
    if wrote_discipline_panels:
        for scenario_name, _guest_roster in scenarios:
            for mode in (DEMOGRAPHIC_MODE_PROPORTIONAL, DEMOGRAPHIC_MODE_WOMEN_TO_PARITY):
                assert (artifact_dir / _discipline_panel_filename(scenario_name, mode)).exists()

    mean_by_mode: dict[str, dict[str, float]] = defaultdict(dict)
    for demographic_mode in (DEMOGRAPHIC_MODE_PROPORTIONAL, DEMOGRAPHIC_MODE_WOMEN_TO_PARITY):
        for scenario_name, _ in scenarios:
            shares = [
                row.female_share
                for row in all_results
                if row.scenario == scenario_name and row.demographic_mode == demographic_mode
            ]
            mean_by_mode[demographic_mode][scenario_name] = mean(shares)

    for scenario_name, _guest_roster in scenarios:
        assert mean_by_mode[DEMOGRAPHIC_MODE_WOMEN_TO_PARITY][scenario_name] >= mean_by_mode[
            DEMOGRAPHIC_MODE_PROPORTIONAL
        ][scenario_name]
