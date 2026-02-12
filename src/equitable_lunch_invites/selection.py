from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from equitable_lunch_invites.models import (
    DISCIPLINES,
    EventPlan,
    Participant,
    PlannerConfig,
    STATE_SCHEMA_VERSION,
)
from equitable_lunch_invites.state import ROLE_GUESTS, ROLE_HOSTS, score_key


def _discipline_order(participants: list[Participant]) -> list[str]:
    available = {participant.discipline for participant in participants}
    ordered = [discipline for discipline in DISCIPLINES if discipline in available]
    extras = sorted(available - set(DISCIPLINES))
    return ordered + extras


def _build_tiebreak(participants: list[Participant], rng: random.Random) -> dict[str, float]:
    return {participant.name: rng.random() for participant in participants}


def _name_sort_key(name: str) -> tuple[int, Any]:
    if name.isdigit():
        return (0, int(name))
    return (1, name.lower())


def allocate_counts_proportional(
    total: int,
    weights: dict[str, int],
    minimums: dict[str, int],
    caps: dict[str, int],
    rng: random.Random,
) -> dict[str, int]:
    keys = list(weights.keys())
    if total <= 0:
        return {key: 0 for key in keys}

    cap_by_key = {key: max(0, int(caps.get(key, total))) for key in keys}
    min_by_key = {
        key: max(0, min(int(minimums.get(key, 0)), cap_by_key[key]))
        for key in keys
    }

    min_total = sum(min_by_key.values())
    alloc = {key: 0 for key in keys}

    if min_total > total:
        # If minimums are oversubscribed, satisfy as many minimum seats as possible.
        order = keys[:]
        rng.shuffle(order)
        order.sort(key=lambda key: (min_by_key[key], weights.get(key, 0)), reverse=True)
        remaining = total
        for key in order:
            take = min(min_by_key[key], cap_by_key[key], remaining)
            alloc[key] = take
            remaining -= take
            if remaining <= 0:
                break
        return alloc

    alloc = min_by_key.copy()
    remaining = total - min_total
    if remaining <= 0:
        return alloc

    positive_weight_sum = sum(max(0, int(weights.get(key, 0))) for key in keys)
    if positive_weight_sum <= 0:
        order = keys[:]
        rng.shuffle(order)
        for key in order:
            if remaining <= 0:
                break
            available = max(0, cap_by_key[key] - alloc[key])
            take = min(available, remaining)
            alloc[key] += take
            remaining -= take
        return alloc

    remainders: list[tuple[float, float, str]] = []
    for key in keys:
        desired = remaining * (max(0, int(weights.get(key, 0))) / positive_weight_sum)
        base = min(int(math.floor(desired)), max(0, cap_by_key[key] - alloc[key]))
        alloc[key] += max(0, base)
        remainders.append((desired - base, rng.random(), key))

    remaining = total - sum(alloc.values())
    remainders.sort(reverse=True)

    idx = 0
    while remaining > 0 and idx < len(remainders):
        _, _, key = remainders[idx]
        if alloc[key] < cap_by_key[key]:
            alloc[key] += 1
            remaining -= 1
        idx += 1

    if remaining > 0:
        order = keys[:]
        rng.shuffle(order)
        for key in order:
            while remaining > 0 and alloc[key] < cap_by_key[key]:
                alloc[key] += 1
                remaining -= 1

    return alloc


def allocate_counts_deterministic(total: int, weights: dict[str, int]) -> dict[str, int]:
    keys = sorted(weights.keys())
    alloc = {key: 0 for key in keys}
    if total <= 0:
        return alloc

    positive_weight_sum = sum(max(0, int(weights[key])) for key in keys)
    if positive_weight_sum <= 0:
        for key in keys:
            if total <= 0:
                break
            alloc[key] += 1
            total -= 1
        return alloc

    exact: dict[str, float] = {}
    for key in keys:
        exact[key] = total * (max(0, int(weights[key])) / positive_weight_sum)
        alloc[key] = int(math.floor(exact[key]))

    remaining = total - sum(alloc.values())
    remainder_order = sorted(keys, key=lambda key: (exact[key] - alloc[key], key), reverse=True)
    idx = 0
    while remaining > 0 and remainder_order:
        key = remainder_order[idx % len(remainder_order)]
        alloc[key] += 1
        remaining -= 1
        idx += 1

    return alloc


def demographic_targets_for_event(
    event_index: int,
    seats_per_event: int,
    demographic_counts: dict[str, int],
) -> dict[str, int]:
    keys = sorted(demographic_counts.keys())
    if event_index <= 0 or seats_per_event <= 0 or not keys:
        return {key: 0 for key in keys}

    cumulative_now = allocate_counts_deterministic(event_index * seats_per_event, demographic_counts)
    cumulative_previous = allocate_counts_deterministic((event_index - 1) * seats_per_event, demographic_counts)
    return {key: cumulative_now[key] - cumulative_previous[key] for key in keys}


def eligible_pool_with_cooldown_override(
    participants: list[Participant],
    role_bucket: dict[str, dict[str, int]],
) -> list[Participant]:
    without_cooldown = [
        participant
        for participant in participants
        if int(role_bucket.get(participant.name, {}).get("cooldown", 0)) <= 0
    ]
    if not participants:
        return []

    all_disciplines = {participant.discipline for participant in participants}
    active_disciplines = {participant.discipline for participant in without_cooldown}
    missing_disciplines = all_disciplines - active_disciplines
    if not missing_disciplines:
        return without_cooldown

    expanded = without_cooldown + [
        participant for participant in participants if participant.discipline in missing_disciplines
    ]
    seen: set[str] = set()
    deduped: list[Participant] = []
    for participant in expanded:
        if participant.name in seen:
            continue
        seen.add(participant.name)
        deduped.append(participant)
    return deduped


def _pick_best_candidate(
    candidates: list[Participant],
    selected_names: set[str],
    role_bucket: dict[str, dict[str, int]],
    tiebreak: dict[str, float],
    demographic_deficits: dict[str, int] | None,
) -> Participant | None:
    pool = [participant for participant in candidates if participant.name not in selected_names]
    if not pool:
        return None

    pool.sort(
        key=lambda participant: _candidate_sort_key(
            participant=participant,
            role_bucket=role_bucket,
            tiebreak=tiebreak,
            demographic_deficits=demographic_deficits,
        )
    )
    return pool[0]


def _candidate_sort_key(
    participant: Participant,
    role_bucket: dict[str, dict[str, int]],
    tiebreak: dict[str, float],
    demographic_deficits: dict[str, int] | None,
) -> tuple[Any, ...]:
    positive_deficits = {
        category: deficit
        for category, deficit in (demographic_deficits or {}).items()
        if deficit > 0
    }

    def candidate_priority(participant: Participant) -> int:
        if not positive_deficits:
            return 0
        demographic = participant.demographic or "U"
        return 0 if positive_deficits.get(demographic, 0) > 0 else 1

    return (
        candidate_priority(participant),
        score_key(role_bucket, participant.name),
        tiebreak.get(participant.name, 0.0),
        _name_sort_key(participant.name),
    )


def _fairness_sorted_waitlist(
    candidates: list[Participant],
    selected_names: set[str],
    role_bucket: dict[str, dict[str, int]],
    tiebreak: dict[str, float],
) -> list[Participant]:
    leftovers = [participant for participant in candidates if participant.name not in selected_names]
    leftovers.sort(
        key=lambda participant: (
            score_key(role_bucket, participant.name),
            tiebreak.get(participant.name, 0.0),
            _name_sort_key(participant.name),
        )
    )
    return leftovers


def select_participants_for_event(
    participants: list[Participant],
    role_bucket: dict[str, dict[str, int]],
    seats: int,
    rng: random.Random,
    waitlist_size: int,
    demographic_targets: dict[str, int] | None = None,
) -> tuple[list[Participant], list[Participant]]:
    if seats <= 0 or not participants:
        return [], []

    selected: list[Participant] = []
    selected_names: set[str] = set()
    selected_demographics = Counter()
    tiebreak = _build_tiebreak(participants, rng)

    def current_deficits() -> dict[str, int]:
        if not demographic_targets:
            return {}
        return {
            category: demographic_targets.get(category, 0) - selected_demographics.get(category, 0)
            for category in demographic_targets
        }

    discipline_order = _discipline_order(participants)
    anchor_slots = min(seats, len(discipline_order))
    anchored_disciplines: set[str] = set()
    for _ in range(anchor_slots):
        deficits = current_deficits()
        discipline_choices: list[tuple[tuple[Any, ...], str, Participant]] = []
        for discipline in discipline_order:
            if discipline in anchored_disciplines:
                continue
            discipline_pool = [
                participant
                for participant in participants
                if participant.discipline == discipline
            ]
            candidate = _pick_best_candidate(
                candidates=discipline_pool,
                selected_names=selected_names,
                role_bucket=role_bucket,
                tiebreak=tiebreak,
                demographic_deficits=deficits,
            )
            if not candidate:
                continue
            discipline_choices.append(
                (
                    _candidate_sort_key(
                        participant=candidate,
                        role_bucket=role_bucket,
                        tiebreak=tiebreak,
                        demographic_deficits=deficits,
                    ),
                    discipline,
                    candidate,
                )
            )

        if not discipline_choices:
            break

        discipline_choices.sort(key=lambda item: item[0])
        _, chosen_discipline, chosen = discipline_choices[0]
        anchored_disciplines.add(chosen_discipline)
        selected.append(chosen)
        selected_names.add(chosen.name)
        selected_demographics[chosen.demographic or "U"] += 1

    while len(selected) < seats:
        chosen = _pick_best_candidate(
            candidates=participants,
            selected_names=selected_names,
            role_bucket=role_bucket,
            tiebreak=tiebreak,
            demographic_deficits=current_deficits(),
        )
        if not chosen:
            break
        selected.append(chosen)
        selected_names.add(chosen.name)
        selected_demographics[chosen.demographic or "U"] += 1

    waitlist = _fairness_sorted_waitlist(
        candidates=participants,
        selected_names=selected_names,
        role_bucket=role_bucket,
        tiebreak=tiebreak,
    )[: max(0, waitlist_size)]

    return selected, waitlist


def choose_guest_cohort(
    roster: list[Participant],
    guest_bucket: dict[str, dict[str, int]],
    max_unique: int,
    cohort_seed: int,
) -> list[str]:
    if not roster:
        return []

    rng = random.Random(cohort_seed)
    cohort_size = min(max_unique, len(roster))
    if cohort_size <= 0:
        return []

    by_discipline: dict[str, list[Participant]] = {}
    for participant in roster:
        by_discipline.setdefault(participant.discipline, []).append(participant)

    disciplines = _discipline_order(roster)
    discipline_sizes = {discipline: len(by_discipline.get(discipline, [])) for discipline in disciplines}

    if cohort_size < len(disciplines):
        discipline_order = disciplines[:]
        rng.shuffle(discipline_order)
        discipline_order.sort(key=lambda discipline: discipline_sizes[discipline], reverse=True)
        included = set(discipline_order[:cohort_size])
        minimums = {discipline: (1 if discipline in included else 0) for discipline in disciplines}
    else:
        base_min = 2 if cohort_size >= 2 * len(disciplines) else 1
        minimums = {
            discipline: min(base_min, discipline_sizes[discipline])
            for discipline in disciplines
        }

    discipline_alloc = allocate_counts_proportional(
        total=cohort_size,
        weights=discipline_sizes,
        minimums=minimums,
        caps=discipline_sizes,
        rng=rng,
    )

    demographic_counts = Counter(participant.demographic or "U" for participant in roster)
    demographic_targets = allocate_counts_proportional(
        total=cohort_size,
        weights={key: int(value) for key, value in demographic_counts.items()},
        minimums={key: 0 for key in demographic_counts},
        caps={key: int(value) for key, value in demographic_counts.items()},
        rng=rng,
    )
    demographic_remaining = demographic_targets.copy()

    tiebreak = _build_tiebreak(roster, rng)
    selected: list[str] = []
    selected_set: set[str] = set()

    def pick_candidate(candidates: list[Participant]) -> Participant | None:
        pool = [participant for participant in candidates if participant.name not in selected_set]
        if not pool:
            return None

        def demographic_priority(participant: Participant) -> int:
            demographic = participant.demographic or "U"
            return 0 if demographic_remaining.get(demographic, 0) > 0 else 1

        pool.sort(
            key=lambda participant: (
                demographic_priority(participant),
                score_key(guest_bucket, participant.name),
                tiebreak.get(participant.name, 0.0),
                _name_sort_key(participant.name),
            )
        )
        return pool[0]

    for discipline in disciplines:
        target_for_discipline = discipline_alloc.get(discipline, 0)
        if target_for_discipline <= 0:
            continue
        candidates = by_discipline.get(discipline, [])
        for _ in range(target_for_discipline):
            candidate = pick_candidate(candidates)
            if not candidate:
                break
            selected.append(candidate.name)
            selected_set.add(candidate.name)
            category = candidate.demographic or "U"
            demographic_remaining[category] = max(0, demographic_remaining.get(category, 0) - 1)

    if len(selected) < cohort_size:
        leftovers = [participant for participant in roster if participant.name not in selected_set]
        while len(selected) < cohort_size and leftovers:
            candidate = pick_candidate(leftovers)
            if not candidate:
                break
            selected.append(candidate.name)
            selected_set.add(candidate.name)
            category = candidate.demographic or "U"
            demographic_remaining[category] = max(0, demographic_remaining.get(category, 0) - 1)
            leftovers = [participant for participant in leftovers if participant.name not in selected_set]

    return selected[:cohort_size]


def _role_buckets(state: dict[str, Any]) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    hosts = state.get(ROLE_HOSTS, {})
    guests = state.get(ROLE_GUESTS, {})
    if not isinstance(hosts, dict) or not isinstance(guests, dict):
        raise ValueError("State roles must be JSON objects keyed by participant name.")
    return hosts, guests


def plan_event(
    host_roster: list[Participant],
    guest_roster: list[Participant],
    state: dict[str, Any],
    config: PlannerConfig,
) -> EventPlan:
    if config.event_index <= 0:
        raise ValueError("--event-index must be >= 1")
    if config.total_events <= 0:
        raise ValueError("--total-events must be >= 1")
    if config.hosts_per_event < 0 or config.guests_per_event < 0:
        raise ValueError("Per-event counts must be >= 0.")
    if config.guest_max_unique <= 0:
        raise ValueError("--guest-max-unique must be >= 1.")

    host_bucket, guest_bucket = _role_buckets(state)
    meta = state.setdefault("_meta", {})
    if not isinstance(meta, dict):
        raise ValueError("State _meta field must be a JSON object.")
    meta["schema_version"] = STATE_SCHEMA_VERSION
    meta["guest_demographic_column"] = config.guest_demographic_column

    guest_lookup = {participant.name: participant for participant in guest_roster}
    guest_cohort = meta.get("guest_cohort", [])
    needs_new_cohort = (
        not isinstance(guest_cohort, list)
        or meta.get("guest_max_unique") != config.guest_max_unique
        or meta.get("cohort_seed") != config.cohort_seed
        or any(name not in guest_lookup for name in guest_cohort)
    )
    if needs_new_cohort:
        guest_cohort = choose_guest_cohort(
            roster=guest_roster,
            guest_bucket=guest_bucket,
            max_unique=config.guest_max_unique,
            cohort_seed=config.cohort_seed,
        )
        meta["guest_cohort"] = guest_cohort
        meta["guest_max_unique"] = config.guest_max_unique
        meta["cohort_seed"] = config.cohort_seed

    cohort_participants = [guest_lookup[name] for name in guest_cohort if name in guest_lookup]
    host_pool = eligible_pool_with_cooldown_override(host_roster, host_bucket)
    guest_pool = eligible_pool_with_cooldown_override(cohort_participants, guest_bucket)

    guest_demographic_counts = Counter(participant.demographic or "U" for participant in guest_roster)
    guest_targets = demographic_targets_for_event(
        event_index=config.event_index,
        seats_per_event=config.guests_per_event,
        demographic_counts={key: int(value) for key, value in guest_demographic_counts.items()},
    )

    host_rng = random.Random((config.seed * 17) + 3)
    guest_rng = random.Random((config.seed * 29) + 7)

    selected_hosts, waitlist_hosts = select_participants_for_event(
        participants=host_pool,
        role_bucket=host_bucket,
        seats=config.hosts_per_event,
        rng=host_rng,
        waitlist_size=config.waitlist_size,
        demographic_targets=None,
    )
    selected_guests, waitlist_guests = select_participants_for_event(
        participants=guest_pool,
        role_bucket=guest_bucket,
        seats=config.guests_per_event,
        rng=guest_rng,
        waitlist_size=config.waitlist_size,
        demographic_targets=guest_targets,
    )

    selected_guest_demographics = Counter(participant.demographic or "U" for participant in selected_guests)
    info: dict[str, Any] = {
        "event_index": config.event_index,
        "total_events": config.total_events,
        "seed": config.seed,
        "cohort_seed": config.cohort_seed,
        "guest_max_unique": config.guest_max_unique,
        "hosts_per_event": config.hosts_per_event,
        "guests_per_event": config.guests_per_event,
        "guest_demographic_column": config.guest_demographic_column,
        "guest_demographic_targets": guest_targets,
        "selected_guest_demographic_counts": dict(selected_guest_demographics),
        "selected_hosts": [participant.name for participant in selected_hosts],
        "selected_guests": [participant.name for participant in selected_guests],
        "host_waitlist": [participant.name for participant in waitlist_hosts],
        "guest_waitlist": [participant.name for participant in waitlist_guests],
        "guest_cohort": guest_cohort,
    }

    history = meta.setdefault("event_history", {})
    if isinstance(history, dict):
        history[str(config.event_index)] = {
            "seed": config.seed,
            "cohort_seed": config.cohort_seed,
            "selected_hosts": info["selected_hosts"],
            "selected_guests": info["selected_guests"],
        }
    meta["last_planned_event"] = config.event_index

    return EventPlan(
        selected_hosts=selected_hosts,
        waitlist_hosts=waitlist_hosts,
        selected_guests=selected_guests,
        waitlist_guests=waitlist_guests,
        info=info,
    )
