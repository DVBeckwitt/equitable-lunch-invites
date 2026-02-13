from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from equitable_lunch_invites.io import (
    PLAN_CSV_HEADER,
    PLAN_ROLE_GUEST,
    PLAN_STATUS_WAITLIST,
    load_guest_roster,
    load_host_roster,
    read_plan_attendance,
    upsert_event_plan_csv,
)
from equitable_lunch_invites.models import (
    DEFAULT_DEMOGRAPHIC_MODE,
    DEFAULT_GUEST_DEMOGRAPHIC_COLUMN,
    DEMOGRAPHIC_MODE_PROPORTIONAL,
    DEMOGRAPHIC_MODE_WOMEN_TO_PARITY,
    PlannerConfig,
)
from equitable_lunch_invites.selection import plan_event
from equitable_lunch_invites.state import (
    ROLE_GUESTS,
    ROLE_HOSTS,
    advance_role_cooldowns,
    apply_role_outcomes,
    default_state,
    ensure_role_records,
    ensure_state_shape,
    load_state,
    save_state,
)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_STATE_PATH = DEFAULT_DATA_DIR / "state.json"
DEFAULT_PLAN_CSV_PATH = DEFAULT_DATA_DIR / "lunch_plan.csv"
DEFAULT_HOST_ROSTER_SHEET = "host_roster"
DEFAULT_GUEST_ROSTER_SHEET = "guest_roster"


def _seed_key_type(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seed-key must be an integer.") from exc
    if value < 0 or value > 1000:
        raise argparse.ArgumentTypeError("--seed-key must be between 0 and 1000.")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="equitable-invites")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser(
        "init",
        help="Initialize planner state from a single XLSX workbook + seed-key.",
    )
    p_init.add_argument("--inputs", required=True, type=Path, help="Workbook .xlsx path.")
    p_init.add_argument("--seed-key", required=True, type=_seed_key_type, help="Integer in [0, 1000].")
    p_init.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    p_init.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV_PATH)
    p_init.add_argument("--host-roster-sheet", default=DEFAULT_HOST_ROSTER_SHEET)
    p_init.add_argument("--guest-roster-sheet", default=DEFAULT_GUEST_ROSTER_SHEET)
    p_init.add_argument(
        "--guest-demographic-column",
        default=DEFAULT_GUEST_DEMOGRAPHIC_COLUMN,
        help=f"Guest demographic column name (default: {DEFAULT_GUEST_DEMOGRAPHIC_COLUMN}).",
    )
    p_init.add_argument(
        "--demographic-mode",
        choices=[DEMOGRAPHIC_MODE_WOMEN_TO_PARITY, DEMOGRAPHIC_MODE_PROPORTIONAL],
        default=DEFAULT_DEMOGRAPHIC_MODE,
        help=(
            "Demographic targeting mode. "
            f"Default: {DEFAULT_DEMOGRAPHIC_MODE}."
        ),
    )

    p_plan = sub.add_parser(
        "plan",
        help="Plan next event and write all rows to one plan CSV (default: data/lunch_plan.csv).",
    )
    p_plan.add_argument("--inputs", required=True, type=Path, help="Workbook .xlsx path.")
    p_plan.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    p_plan.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV_PATH)
    p_plan.add_argument("--event-index", type=int, help="Optional explicit event index.")
    p_plan.add_argument("--hosts-per-event", type=int, default=2)
    p_plan.add_argument("--guests-per-event", type=int, default=5)
    p_plan.add_argument("--total-events", type=int, default=5)
    p_plan.add_argument(
        "--guest-max-unique",
        type=int,
        default=None,
        help="Max unique guests across the series (default: total guests in guest roster).",
    )
    p_plan.add_argument("--waitlist-size", type=int, default=8)
    p_plan.add_argument("--host-roster-sheet", default=DEFAULT_HOST_ROSTER_SHEET)
    p_plan.add_argument("--guest-roster-sheet", default=DEFAULT_GUEST_ROSTER_SHEET)
    p_plan.add_argument(
        "--guest-demographic-column",
        default=DEFAULT_GUEST_DEMOGRAPHIC_COLUMN,
        help=f"Guest demographic column name (default: {DEFAULT_GUEST_DEMOGRAPHIC_COLUMN}).",
    )
    p_plan.add_argument(
        "--demographic-mode",
        choices=[DEMOGRAPHIC_MODE_WOMEN_TO_PARITY, DEMOGRAPHIC_MODE_PROPORTIONAL],
        default=DEFAULT_DEMOGRAPHIC_MODE,
        help=(
            "Demographic targeting mode. "
            f"Default: {DEFAULT_DEMOGRAPHIC_MODE}."
        ),
    )

    p_start = sub.add_parser(
        "start",
        help="Initialize and plan in one command.",
    )
    p_start.add_argument("--inputs", required=True, type=Path, help="Workbook .xlsx path.")
    p_start.add_argument("--seed-key", required=True, type=_seed_key_type, help="Integer in [0, 1000].")
    p_start.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    p_start.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV_PATH)
    p_start.add_argument("--event-index", type=int, help="Optional explicit event index.")
    p_start.add_argument("--hosts-per-event", type=int, default=2)
    p_start.add_argument("--guests-per-event", type=int, default=5)
    p_start.add_argument("--total-events", type=int, default=5)
    p_start.add_argument(
        "--guest-max-unique",
        type=int,
        default=None,
        help="Max unique guests across the series (default: total guests in guest roster).",
    )
    p_start.add_argument("--waitlist-size", type=int, default=8)
    p_start.add_argument("--host-roster-sheet", default=DEFAULT_HOST_ROSTER_SHEET)
    p_start.add_argument("--guest-roster-sheet", default=DEFAULT_GUEST_ROSTER_SHEET)
    p_start.add_argument(
        "--guest-demographic-column",
        default=DEFAULT_GUEST_DEMOGRAPHIC_COLUMN,
        help=f"Guest demographic column name (default: {DEFAULT_GUEST_DEMOGRAPHIC_COLUMN}).",
    )
    p_start.add_argument(
        "--demographic-mode",
        choices=[DEMOGRAPHIC_MODE_WOMEN_TO_PARITY, DEMOGRAPHIC_MODE_PROPORTIONAL],
        default=DEFAULT_DEMOGRAPHIC_MODE,
        help=(
            "Demographic targeting mode. "
            f"Default: {DEFAULT_DEMOGRAPHIC_MODE}."
        ),
    )

    return parser


def _validate_workbook_path(path: Path) -> None:
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Expected a single .xlsx workbook, got: {path}")


def _derive_seed(seed_key: int, purpose: str) -> int:
    digest = hashlib.sha256(f"{seed_key}:{purpose}".encode("utf-8")).digest()
    # Keep seeds in a typical signed 32-bit RNG range.
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def _load_workbook_rosters(args: argparse.Namespace) -> tuple[list, list]:
    _validate_workbook_path(args.inputs)
    host_roster = load_host_roster(args.inputs, sheet_name=args.host_roster_sheet)
    guest_roster = load_guest_roster(
        args.inputs,
        demographic_column=args.guest_demographic_column,
        outcome_column=None,
        sheet_name=args.guest_roster_sheet,
    )
    return host_roster, guest_roster


def _ensure_plan_csv_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_CSV_HEADER)
        writer.writeheader()


def _selected_names_for_event(state: dict[str, Any], event_index: int) -> tuple[list[str], list[str]]:
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


def _record_event_from_plan_csv(
    state: dict[str, Any],
    plan_csv: Path,
    event_index: int,
) -> None:
    selected_hosts, selected_guests = _selected_names_for_event(state, event_index)

    # No-show cooldown applies only to guests.
    advance_role_cooldowns(state, ROLE_GUESTS)

    if selected_hosts:
        apply_role_outcomes(state, ROLE_HOSTS, attended=selected_hosts, no_show=[])

    selected_attendance_map = read_plan_attendance(
        plan_csv,
        event_index=event_index,
        role=PLAN_ROLE_GUEST,
    )
    waitlist_attendance_map = read_plan_attendance(
        plan_csv,
        event_index=event_index,
        role=PLAN_ROLE_GUEST,
        status=PLAN_STATUS_WAITLIST,
    )
    missing: list[str] = []
    attended: list[str] = []
    no_show: list[str] = []
    cant_attend: list[str] = []
    for name in selected_guests:
        outcome = selected_attendance_map.get(name, "")
        if outcome == "attended":
            attended.append(name)
        elif outcome == "no_show":
            no_show.append(name)
        elif outcome == "cant_attend":
            cant_attend.append(name)
        else:
            missing.append(name)

    # Waitlist guests may fill-in for selected guests.
    for name, outcome in waitlist_attendance_map.items():
        if outcome in {"filled", "attended"}:
            attended.append(name)
        elif outcome == "no_show":
            no_show.append(name)
        elif outcome in {"cant_attend", ""}:
            continue

    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(
            f"Missing guest attendance in {plan_csv} for event {event_index}: {preview}. "
            "Fill the 'attendance' column with attended/no_show/cant_attend for selected guests."
        )

    apply_role_outcomes(
        state,
        ROLE_GUESTS,
        attended=attended,
        no_show=no_show,
        cant_attend=cant_attend,
    )
    state["_meta"]["last_recorded_event"] = event_index


def _record_pending_events_before(
    state: dict[str, Any],
    plan_csv: Path,
    target_event_index: int,
) -> list[int]:
    if target_event_index <= 1:
        return []

    meta = state.get("_meta", {})
    if not isinstance(meta, dict):
        raise ValueError("State _meta field must be a JSON object.")
    last_recorded = int(meta.get("last_recorded_event", 0) or 0)
    if last_recorded >= target_event_index - 1:
        return []

    recorded_now: list[int] = []
    for event_index in range(last_recorded + 1, target_event_index):
        _record_event_from_plan_csv(state, plan_csv, event_index)
        recorded_now.append(event_index)
    return recorded_now


def _event_index_for_plan(meta: dict[str, Any], explicit_event: int | None) -> int:
    if explicit_event is not None:
        if explicit_event <= 0:
            raise ValueError("--event-index must be >= 1.")
        return explicit_event
    return int(meta.get("last_planned_event", 0) or 0) + 1


def run_init(args: argparse.Namespace) -> int:
    host_roster, guest_roster = _load_workbook_rosters(args)

    state = default_state()
    ensure_state_shape(state)
    ensure_role_records(state, ROLE_HOSTS, [participant.name for participant in host_roster])
    ensure_role_records(state, ROLE_GUESTS, [participant.name for participant in guest_roster])
    state["_meta"]["seed_key"] = args.seed_key
    state["_meta"]["generated_seed"] = _derive_seed(args.seed_key, "base")
    state["_meta"]["inputs"] = str(args.inputs)
    state["_meta"]["plan_csv"] = str(args.plan_csv)
    state["_meta"]["host_roster_sheet"] = args.host_roster_sheet
    state["_meta"]["guest_roster_sheet"] = args.guest_roster_sheet
    state["_meta"]["guest_demographic_column"] = args.guest_demographic_column
    state["_meta"]["demographic_mode"] = args.demographic_mode
    save_state(args.state, state)
    _ensure_plan_csv_header(args.plan_csv)

    print(
        f"Initialized {args.state} using seed-key {args.seed_key} "
        f"for {len(host_roster)} hosts and {len(guest_roster)} guests. "
        f"Plan CSV: {args.plan_csv}"
    )
    return 0


def run_plan(args: argparse.Namespace) -> int:
    host_roster, guest_roster = _load_workbook_rosters(args)

    state = load_state(args.state)
    ensure_state_shape(state)
    ensure_role_records(state, ROLE_HOSTS, [participant.name for participant in host_roster])
    ensure_role_records(state, ROLE_GUESTS, [participant.name for participant in guest_roster])

    meta = state.get("_meta", {})
    if not isinstance(meta, dict):
        raise ValueError("State _meta field must be a JSON object.")
    if "seed_key" not in meta:
        raise ValueError(f"State {args.state} is not initialized. Run init first.")

    event_index = _event_index_for_plan(meta, args.event_index)
    last_recorded_event = int(meta.get("last_recorded_event", 0) or 0)
    if event_index <= last_recorded_event:
        raise ValueError(
            f"Event {event_index} is already recorded (last recorded: {last_recorded_event})."
        )

    recorded_now = _record_pending_events_before(
        state=state,
        plan_csv=args.plan_csv,
        target_event_index=event_index,
    )

    seed_key = int(meta["seed_key"])
    event_seed = _derive_seed(seed_key, f"event:{event_index}")
    cohort_seed = _derive_seed(seed_key, "cohort")

    guest_max_unique = (
        args.guest_max_unique
        if args.guest_max_unique is not None
        else len(guest_roster)
    )
    config = PlannerConfig(
        event_index=event_index,
        total_events=args.total_events,
        hosts_per_event=args.hosts_per_event,
        guests_per_event=args.guests_per_event,
        guest_max_unique=guest_max_unique,
        seed=event_seed,
        cohort_seed=cohort_seed,
        guest_demographic_column=args.guest_demographic_column,
        demographic_mode=args.demographic_mode,
        waitlist_size=args.waitlist_size,
    )
    event_plan = plan_event(
        host_roster=host_roster,
        guest_roster=guest_roster,
        state=state,
        config=config,
    )

    upsert_event_plan_csv(
        path=args.plan_csv,
        event_index=event_index,
        selected_hosts=event_plan.selected_hosts,
        waitlist_hosts=event_plan.waitlist_hosts,
        selected_guests=event_plan.selected_guests,
        waitlist_guests=event_plan.waitlist_guests,
    )
    save_state(args.state, state)

    info = dict(event_plan.info)
    info["seed_key"] = seed_key
    info["generated_event_seed"] = event_seed
    info["generated_cohort_seed"] = cohort_seed
    info["recorded_events_before_planning"] = recorded_now
    info["plan_csv"] = str(args.plan_csv)
    print(json.dumps(info, indent=2))
    return 0


def run_start(args: argparse.Namespace) -> int:
    run_init(args)
    return run_plan(args)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "init":
        return run_init(args)
    if args.cmd == "plan":
        return run_plan(args)
    if args.cmd == "start":
        return run_start(args)

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
