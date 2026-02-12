#!/usr/bin/env python3
"""
Legacy compatibility wrapper for the old single-roster lunch CLI.

This script preserves these legacy commands:
  python lunches.py init --roster roster.csv --state state.json
  python lunches.py plan --roster roster.csv --state state.json --lunch-index 1 --seed 123 --out out.csv --wait wait.csv
  python lunches.py record --state state.json --attended attended.csv --no_show no_show.csv --lunch-index 1

Internally, legacy commands are mapped onto the new role-aware planner
with guest-role defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from equitable_lunch_invites.io import (
    build_numbered_guests,
    load_guest_roster,
    read_csv_firstcol,
    write_participant_list,
)
from equitable_lunch_invites.models import PlannerConfig
from equitable_lunch_invites.selection import plan_event
from equitable_lunch_invites.state import (
    ROLE_GUESTS,
    ROLE_HOSTS,
    advance_role_cooldowns,
    apply_role_outcomes,
    ensure_role_records,
    ensure_state_shape,
    load_state,
    save_state,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lunches.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    init_guest = p_init.add_mutually_exclusive_group(required=True)
    init_guest.add_argument("--roster", type=Path)
    init_guest.add_argument("--guest-total", type=int)
    p_init.add_argument("--state", required=True, type=Path)

    p_plan = sub.add_parser("plan")
    plan_guest = p_plan.add_mutually_exclusive_group(required=True)
    plan_guest.add_argument("--roster", type=Path)
    plan_guest.add_argument("--guest-total", type=int)
    p_plan.add_argument("--state", required=True, type=Path)
    p_plan.add_argument("--lunch-index", required=True, type=int)
    p_plan.add_argument("--seed", required=True, type=int)
    p_plan.add_argument("--out", required=True, type=Path)
    p_plan.add_argument("--wait", required=True, type=Path)
    p_plan.add_argument("--seats", type=int, default=5)
    p_plan.add_argument("--total-lunches", type=int, default=5)
    p_plan.add_argument("--max-unique", type=int, default=10)
    p_plan.add_argument("--cohort-seed", type=int, default=777)

    p_record = sub.add_parser("record")
    p_record.add_argument("--state", required=True, type=Path)
    p_record.add_argument("--attended", required=True, type=Path)
    p_record.add_argument("--no_show", required=True, type=Path)
    p_record.add_argument("--lunch-index", required=True, type=int)

    return parser


def _load_legacy_guests(args: argparse.Namespace) -> tuple[list, bool]:
    if args.guest_total is not None:
        return build_numbered_guests(args.guest_total), True
    if args.roster is None:
        raise ValueError("Either --roster or --guest-total is required.")
    return load_guest_roster(args.roster, demographic_column="Sex"), False


def _run_legacy_init(args: argparse.Namespace) -> int:
    guest_roster, guest_is_anonymous = _load_legacy_guests(args)
    state = load_state(args.state)
    ensure_state_shape(state)
    ensure_role_records(state, ROLE_GUESTS, [participant.name for participant in guest_roster])
    state.setdefault(ROLE_HOSTS, {})
    state["_meta"]["guest_demographic_column"] = ("Sex" if not guest_is_anonymous else "Anonymous")
    save_state(args.state, state)
    print(f"Initialized state for {len(guest_roster)} guests at {args.state}")
    return 0


def _run_legacy_plan(args: argparse.Namespace) -> int:
    guest_roster, guest_is_anonymous = _load_legacy_guests(args)
    state = load_state(args.state)
    ensure_state_shape(state)
    ensure_role_records(state, ROLE_GUESTS, [participant.name for participant in guest_roster])

    config = PlannerConfig(
        event_index=args.lunch_index,
        total_events=args.total_lunches,
        hosts_per_event=0,
        guests_per_event=args.seats,
        guest_max_unique=args.max_unique,
        seed=args.seed,
        cohort_seed=args.cohort_seed,
        guest_demographic_column=("Sex" if not guest_is_anonymous else "Anonymous"),
    )

    event_plan = plan_event(
        host_roster=[],
        guest_roster=guest_roster,
        state=state,
        config=config,
    )

    write_participant_list(
        args.out,
        event_plan.selected_guests,
        demographic_header=(None if guest_is_anonymous else "sex"),
    )
    write_participant_list(
        args.wait,
        event_plan.waitlist_guests,
        demographic_header=(None if guest_is_anonymous else "sex"),
    )
    save_state(args.state, state)

    legacy_info = {
        "seed": args.seed,
        "cohort_seed": args.cohort_seed,
        "max_unique": args.max_unique,
        "seats_per_lunch": args.seats,
        "total_lunches": args.total_lunches,
        "selected_names": [participant.name for participant in event_plan.selected_guests],
        "selected_disciplines": [participant.discipline for participant in event_plan.selected_guests],
        "selected_women": [
            participant.name
            for participant in event_plan.selected_guests
            if (participant.demographic or "").upper() == "F"
        ],
        "cohort_names": state.get("_meta", {}).get("guest_cohort", []),
    }
    print(json.dumps(legacy_info, indent=2))
    return 0


def _run_legacy_record(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    ensure_state_shape(state)
    advance_role_cooldowns(state, ROLE_HOSTS)
    advance_role_cooldowns(state, ROLE_GUESTS)

    attended_names = read_csv_firstcol(args.attended)
    no_show_names = read_csv_firstcol(args.no_show)
    apply_role_outcomes(state, ROLE_GUESTS, attended=attended_names, no_show=no_show_names)
    state["_meta"]["last_recorded_event"] = args.lunch_index
    save_state(args.state, state)
    print(f"Recorded outcomes for lunch {args.lunch_index} and updated {args.state}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "init":
        return _run_legacy_init(args)
    if args.cmd == "plan":
        return _run_legacy_plan(args)
    if args.cmd == "record":
        return _run_legacy_record(args)
    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
