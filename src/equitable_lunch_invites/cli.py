from __future__ import annotations

import argparse
import json
from pathlib import Path

from equitable_lunch_invites.io import (
    load_guest_roster,
    load_host_roster,
    read_guest_outcomes,
    write_participant_list,
)
from equitable_lunch_invites.models import (
    DEFAULT_GUEST_DEMOGRAPHIC_COLUMN,
    DEFAULT_GUEST_OUTCOME_COLUMN,
    PlannerConfig,
)
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
    parser = argparse.ArgumentParser(prog="equitable-invites")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser(
        "init",
        help="Initialize state from host and guest rosters.",
    )
    p_init.add_argument("--host-roster", required=True, type=Path, help="Host roster CSV/XLSX.")
    p_init.add_argument("--host-roster-sheet", type=str, help="Sheet name when --host-roster is .xlsx.")
    p_init.add_argument("--guest-roster", required=True, type=Path, help="Guest roster CSV/XLSX.")
    p_init.add_argument("--guest-roster-sheet", type=str, help="Sheet name when --guest-roster is .xlsx.")
    p_init.add_argument("--state", required=True, type=Path)
    p_init.add_argument(
        "--guest-demographic-column",
        default=DEFAULT_GUEST_DEMOGRAPHIC_COLUMN,
        help=f"Guest demographic column name (default: {DEFAULT_GUEST_DEMOGRAPHIC_COLUMN}).",
    )
    p_init.add_argument(
        "--guest-outcome-column",
        default=DEFAULT_GUEST_OUTCOME_COLUMN,
        help=f"Guest outcome column name (default: {DEFAULT_GUEST_OUTCOME_COLUMN}).",
    )

    p_plan = sub.add_parser(
        "plan",
        help="Plan one event and write host/guest selections + waitlists.",
    )
    p_plan.add_argument("--host-roster", required=True, type=Path, help="Host roster CSV/XLSX.")
    p_plan.add_argument("--host-roster-sheet", type=str, help="Sheet name when --host-roster is .xlsx.")
    p_plan.add_argument("--guest-roster", required=True, type=Path, help="Guest roster CSV/XLSX.")
    p_plan.add_argument("--guest-roster-sheet", type=str, help="Sheet name when --guest-roster is .xlsx.")
    p_plan.add_argument("--state", required=True, type=Path)
    p_plan.add_argument("--event-index", required=True, type=int)
    p_plan.add_argument("--seed", required=True, type=int)
    p_plan.add_argument("--hosts-per-event", required=True, type=int)
    p_plan.add_argument("--guests-per-event", required=True, type=int)
    p_plan.add_argument("--host-out", required=True, type=Path)
    p_plan.add_argument("--host-wait", required=True, type=Path)
    p_plan.add_argument("--guest-out", required=True, type=Path)
    p_plan.add_argument("--guest-wait", required=True, type=Path)
    p_plan.add_argument("--total-events", type=int, default=5)
    p_plan.add_argument(
        "--guest-max-unique",
        type=int,
        default=None,
        help="Max unique guests across the series (default: total guests in guest roster).",
    )
    p_plan.add_argument("--cohort-seed", type=int, default=777)
    p_plan.add_argument("--waitlist-size", type=int, default=8)
    p_plan.add_argument(
        "--guest-demographic-column",
        default=DEFAULT_GUEST_DEMOGRAPHIC_COLUMN,
        help=f"Guest demographic column name (default: {DEFAULT_GUEST_DEMOGRAPHIC_COLUMN}).",
    )
    p_plan.add_argument(
        "--guest-outcome-column",
        default=DEFAULT_GUEST_OUTCOME_COLUMN,
        help=f"Guest outcome column name (default: {DEFAULT_GUEST_OUTCOME_COLUMN}).",
    )

    p_record = sub.add_parser(
        "record",
        help="Record outcomes from guest roster outcome column; hosts are auto-marked attended.",
    )
    p_record.add_argument("--state", required=True, type=Path)
    p_record.add_argument("--event-index", required=True, type=int)
    p_record.add_argument("--guest-roster", required=True, type=Path, help="Guest roster CSV/XLSX.")
    p_record.add_argument("--guest-roster-sheet", type=str, help="Sheet name when --guest-roster is .xlsx.")
    p_record.add_argument(
        "--guest-outcome-column",
        default=DEFAULT_GUEST_OUTCOME_COLUMN,
        help=f"Guest outcome column name (default: {DEFAULT_GUEST_OUTCOME_COLUMN}).",
    )

    return parser


def _load_host_guest_rosters(args: argparse.Namespace) -> tuple[list, list]:
    host_roster = load_host_roster(
        args.host_roster,
        sheet_name=args.host_roster_sheet,
    )
    guest_roster = load_guest_roster(
        args.guest_roster,
        demographic_column=args.guest_demographic_column,
        outcome_column=args.guest_outcome_column,
        sheet_name=args.guest_roster_sheet,
    )
    return host_roster, guest_roster


def _selected_names_for_event(state: dict, event_index: int) -> tuple[list[str], list[str]]:
    meta = state.get("_meta", {})
    if not isinstance(meta, dict):
        raise ValueError("State _meta field must be a JSON object.")
    history = meta.get("event_history", {})
    if not isinstance(history, dict):
        raise ValueError("State _meta.event_history must be a JSON object.")

    event_entry = history.get(str(event_index))
    if not isinstance(event_entry, dict):
        raise ValueError(
            f"No planned event found for event index {event_index}. Run plan first."
        )

    selected_hosts = event_entry.get("selected_hosts", [])
    selected_guests = event_entry.get("selected_guests", [])
    if not isinstance(selected_hosts, list) or not isinstance(selected_guests, list):
        raise ValueError(f"Invalid event history data for event index {event_index}.")
    return [str(name) for name in selected_hosts], [str(name) for name in selected_guests]


def run_init(args: argparse.Namespace) -> int:
    host_roster, guest_roster = _load_host_guest_rosters(args)

    state = load_state(args.state)
    ensure_state_shape(state)
    ensure_role_records(state, ROLE_HOSTS, [participant.name for participant in host_roster])
    ensure_role_records(state, ROLE_GUESTS, [participant.name for participant in guest_roster])
    state["_meta"]["guest_demographic_column"] = args.guest_demographic_column
    state["_meta"]["guest_outcome_column"] = args.guest_outcome_column
    state["_meta"]["guest_input_mode"] = "roster"
    save_state(args.state, state)

    print(
        f"Initialized state at {args.state} "
        f"for {len(host_roster)} hosts and {len(guest_roster)} guests."
    )
    return 0


def run_plan(args: argparse.Namespace) -> int:
    host_roster, guest_roster = _load_host_guest_rosters(args)

    state = load_state(args.state)
    ensure_state_shape(state)
    ensure_role_records(state, ROLE_HOSTS, [participant.name for participant in host_roster])
    ensure_role_records(state, ROLE_GUESTS, [participant.name for participant in guest_roster])

    guest_max_unique = args.guest_max_unique if args.guest_max_unique is not None else len(guest_roster)
    config = PlannerConfig(
        event_index=args.event_index,
        total_events=args.total_events,
        hosts_per_event=args.hosts_per_event,
        guests_per_event=args.guests_per_event,
        guest_max_unique=guest_max_unique,
        seed=args.seed,
        cohort_seed=args.cohort_seed,
        guest_demographic_column=args.guest_demographic_column,
        waitlist_size=args.waitlist_size,
    )
    event_plan = plan_event(
        host_roster=host_roster,
        guest_roster=guest_roster,
        state=state,
        config=config,
    )

    write_participant_list(args.host_out, event_plan.selected_hosts)
    write_participant_list(args.host_wait, event_plan.waitlist_hosts)
    write_participant_list(
        args.guest_out,
        event_plan.selected_guests,
        demographic_header=args.guest_demographic_column,
    )
    write_participant_list(
        args.guest_wait,
        event_plan.waitlist_guests,
        demographic_header=args.guest_demographic_column,
    )
    save_state(args.state, state)

    print(json.dumps(event_plan.info, indent=2))
    return 0


def run_record(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    ensure_state_shape(state)

    selected_hosts, selected_guests = _selected_names_for_event(state, args.event_index)

    # No-show cooldown applies only to guests in this workflow.
    advance_role_cooldowns(state, ROLE_GUESTS)

    if selected_hosts:
        apply_role_outcomes(state, ROLE_HOSTS, attended=selected_hosts, no_show=[])

    outcome_map = read_guest_outcomes(
        args.guest_roster,
        outcome_column=args.guest_outcome_column,
        sheet_name=args.guest_roster_sheet,
    )
    guest_attended: list[str] = []
    guest_no_show: list[str] = []
    missing: list[str] = []

    for name in selected_guests:
        outcome = outcome_map.get(name)
        if outcome == "attended":
            guest_attended.append(name)
        elif outcome == "no_show":
            guest_no_show.append(name)
        else:
            missing.append(name)

    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(
            f"Missing guest outcomes for selected guests: {preview}. "
            f"Fill '{args.guest_outcome_column}' with attended/no_show for selected guests."
        )

    apply_role_outcomes(state, ROLE_GUESTS, attended=guest_attended, no_show=guest_no_show)
    state["_meta"]["last_recorded_event"] = args.event_index
    save_state(args.state, state)

    print(f"Recorded outcomes for event {args.event_index} in {args.state}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "init":
        return run_init(args)
    if args.cmd == "plan":
        return run_plan(args)
    if args.cmd == "record":
        return run_record(args)

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
