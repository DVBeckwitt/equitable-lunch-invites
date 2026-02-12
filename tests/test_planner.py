from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from equitable_lunch_invites.cli import main as cli_main


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def read_column(path: Path, column: str) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row[column] for row in reader if row.get(column)]


def run_cli(args: list[str]) -> None:
    exit_code = cli_main(args)
    assert exit_code == 0


def make_host_roster(path: Path) -> None:
    rows = [
        ["Host A", "biophysics"],
        ["Host B", "astronomy"],
        ["Host C", "condensed matter experimental"],
        ["Host D", "condensed matter theory"],
        ["Host E", "biophysics"],
        ["Host F", "astronomy"],
    ]
    write_csv(path, ["Name", "Discipline"], rows)


def make_guest_roster(path: Path, demographic_column: str = "Sex") -> None:
    rows = [
        ["Guest A", "biophysics", "F", ""],
        ["Guest B", "astronomy", "M", ""],
        ["Guest C", "condensed matter experimental", "F", ""],
        ["Guest D", "condensed matter theory", "M", ""],
        ["Guest E", "biophysics", "F", ""],
        ["Guest F", "astronomy", "M", ""],
        ["Guest G", "condensed matter experimental", "F", ""],
        ["Guest H", "condensed matter theory", "M", ""],
        ["Guest I", "biophysics", "F", ""],
        ["Guest J", "astronomy", "U", ""],
    ]
    write_csv(path, ["Name", "Discipline", demographic_column, "Outcome"], rows)


def set_guest_outcomes_csv(
    path: Path,
    attended: list[str],
    no_show: list[str],
    outcome_column: str = "Outcome",
) -> None:
    attended_set = set(attended)
    no_show_set = set(no_show)
    overlap = attended_set & no_show_set
    if overlap:
        raise ValueError(f"Guest names cannot be both attended and no_show: {sorted(overlap)}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "Name" not in fieldnames:
        raise ValueError("Guest roster must include Name column.")
    if outcome_column not in fieldnames:
        raise ValueError(f"Guest roster must include {outcome_column} column.")

    known_names = {row["Name"] for row in rows if row.get("Name")}
    unknown = sorted((attended_set | no_show_set) - known_names)
    if unknown:
        raise ValueError(f"Unknown guest names for outcome update: {unknown}")

    for row in rows:
        name = (row.get("Name") or "").strip()
        if name in attended_set:
            row[outcome_column] = "attended"
        elif name in no_show_set:
            row[outcome_column] = "no_show"
        else:
            row[outcome_column] = ""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_plan_selects_counts_and_is_deterministic(tmp_path: Path) -> None:
    host_roster = tmp_path / "hosts.csv"
    guest_roster = tmp_path / "guests.csv"
    state_path = tmp_path / "state.json"
    make_host_roster(host_roster)
    make_guest_roster(guest_roster)

    run_cli(
        [
            "init",
            "--host-roster",
            str(host_roster),
            "--guest-roster",
            str(guest_roster),
            "--state",
            str(state_path),
        ]
    )

    host_out_1 = tmp_path / "host_out_1.csv"
    host_wait_1 = tmp_path / "host_wait_1.csv"
    guest_out_1 = tmp_path / "guest_out_1.csv"
    guest_wait_1 = tmp_path / "guest_wait_1.csv"
    plan_args = [
        "plan",
        "--host-roster",
        str(host_roster),
        "--guest-roster",
        str(guest_roster),
        "--state",
        str(state_path),
        "--event-index",
        "1",
        "--seed",
        "12345",
        "--cohort-seed",
        "777",
        "--hosts-per-event",
        "2",
        "--guests-per-event",
        "5",
        "--guest-max-unique",
        "8",
        "--total-events",
        "5",
        "--host-out",
        str(host_out_1),
        "--host-wait",
        str(host_wait_1),
        "--guest-out",
        str(guest_out_1),
        "--guest-wait",
        str(guest_wait_1),
    ]
    run_cli(plan_args)

    hosts_1 = read_column(host_out_1, "name")
    guests_1 = read_column(guest_out_1, "name")
    assert len(hosts_1) == 2
    assert len(guests_1) == 5

    host_out_2 = tmp_path / "host_out_2.csv"
    host_wait_2 = tmp_path / "host_wait_2.csv"
    guest_out_2 = tmp_path / "guest_out_2.csv"
    guest_wait_2 = tmp_path / "guest_wait_2.csv"
    run_cli(
        plan_args[:-8]
        + [
            "--host-out",
            str(host_out_2),
            "--host-wait",
            str(host_wait_2),
            "--guest-out",
            str(guest_out_2),
            "--guest-wait",
            str(guest_wait_2),
        ]
    )

    assert hosts_1 == read_column(host_out_2, "name")
    assert guests_1 == read_column(guest_out_2, "name")


def test_guest_unique_cap_across_events(tmp_path: Path) -> None:
    host_roster = tmp_path / "hosts.csv"
    guest_roster = tmp_path / "guests.csv"
    state_path = tmp_path / "state.json"
    make_host_roster(host_roster)
    make_guest_roster(guest_roster)

    run_cli(
        [
            "init",
            "--host-roster",
            str(host_roster),
            "--guest-roster",
            str(guest_roster),
            "--state",
            str(state_path),
        ]
    )

    all_selected: set[str] = set()
    for event_index in (1, 2):
        guest_out = tmp_path / f"guest_out_{event_index}.csv"
        run_cli(
            [
                "plan",
                "--host-roster",
                str(host_roster),
                "--guest-roster",
                str(guest_roster),
                "--state",
                str(state_path),
                "--event-index",
                str(event_index),
                "--seed",
                str(2000 + event_index),
                "--cohort-seed",
                "123",
                "--hosts-per-event",
                "1",
                "--guests-per-event",
                "3",
                "--guest-max-unique",
                "4",
                "--total-events",
                "5",
                "--host-out",
                str(tmp_path / f"host_out_{event_index}.csv"),
                "--host-wait",
                str(tmp_path / f"host_wait_{event_index}.csv"),
                "--guest-out",
                str(guest_out),
                "--guest-wait",
                str(tmp_path / f"guest_wait_{event_index}.csv"),
            ]
        )
        all_selected.update(read_column(guest_out, "name"))

    assert len(all_selected) <= 4


def test_hosts_auto_attend_and_guest_outcomes_from_roster(tmp_path: Path) -> None:
    host_roster = tmp_path / "hosts.csv"
    guest_roster = tmp_path / "guests.csv"
    state_path = tmp_path / "state.json"
    make_host_roster(host_roster)
    make_guest_roster(guest_roster)

    run_cli(
        [
            "init",
            "--host-roster",
            str(host_roster),
            "--guest-roster",
            str(guest_roster),
            "--state",
            str(state_path),
        ]
    )

    host_out = tmp_path / "host_out.csv"
    guest_out = tmp_path / "guest_out.csv"
    run_cli(
        [
            "plan",
            "--host-roster",
            str(host_roster),
            "--guest-roster",
            str(guest_roster),
            "--state",
            str(state_path),
            "--event-index",
            "1",
            "--seed",
            "88",
            "--cohort-seed",
            "11",
            "--hosts-per-event",
            "2",
            "--guests-per-event",
            "3",
            "--total-events",
            "5",
            "--host-out",
            str(host_out),
            "--host-wait",
            str(tmp_path / "host_wait.csv"),
            "--guest-out",
            str(guest_out),
            "--guest-wait",
            str(tmp_path / "guest_wait.csv"),
        ]
    )

    selected_hosts = read_column(host_out, "name")
    selected_guests = read_column(guest_out, "name")
    set_guest_outcomes_csv(guest_roster, attended=selected_guests[:2], no_show=[selected_guests[2]])

    run_cli(
        [
            "record",
            "--state",
            str(state_path),
            "--event-index",
            "1",
            "--guest-roster",
            str(guest_roster),
        ]
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    for host_name in selected_hosts:
        assert state["hosts"][host_name]["assigned_count"] == 1
    assert state["guests"][selected_guests[2]]["no_show_count"] == 1
    assert state["guests"][selected_guests[2]]["cooldown"] == 1


def test_cooldown_override_preserves_coverage(tmp_path: Path) -> None:
    host_roster = tmp_path / "hosts.csv"
    guest_roster = tmp_path / "guests.csv"
    state_path = tmp_path / "state.json"
    write_csv(host_roster, ["Name", "Discipline"], [["Host One", "biophysics"]])
    write_csv(
        guest_roster,
        ["Name", "Discipline", "Sex", "Outcome"],
        [
            ["Unique Astro", "astronomy", "F", ""],
            ["Bio 1", "biophysics", "M", ""],
            ["CME 1", "condensed matter experimental", "F", ""],
            ["CMT 1", "condensed matter theory", "M", ""],
            ["Bio 2", "biophysics", "F", ""],
            ["CME 2", "condensed matter experimental", "M", ""],
        ],
    )

    run_cli(
        [
            "init",
            "--host-roster",
            str(host_roster),
            "--guest-roster",
            str(guest_roster),
            "--state",
            str(state_path),
        ]
    )

    guest_out_1 = tmp_path / "guest_out_1.csv"
    run_cli(
        [
            "plan",
            "--host-roster",
            str(host_roster),
            "--guest-roster",
            str(guest_roster),
            "--state",
            str(state_path),
            "--event-index",
            "1",
            "--seed",
            "11",
            "--cohort-seed",
            "22",
            "--hosts-per-event",
            "0",
            "--guests-per-event",
            "4",
            "--guest-max-unique",
            "6",
            "--total-events",
            "2",
            "--host-out",
            str(tmp_path / "host_out_1.csv"),
            "--host-wait",
            str(tmp_path / "host_wait_1.csv"),
            "--guest-out",
            str(guest_out_1),
            "--guest-wait",
            str(tmp_path / "guest_wait_1.csv"),
        ]
    )

    selected_event_1 = read_column(guest_out_1, "name")
    assert "Unique Astro" in selected_event_1

    attended_guests = [name for name in selected_event_1 if name != "Unique Astro"]
    set_guest_outcomes_csv(guest_roster, attended=attended_guests, no_show=["Unique Astro"])
    run_cli(
        [
            "record",
            "--state",
            str(state_path),
            "--event-index",
            "1",
            "--guest-roster",
            str(guest_roster),
        ]
    )

    guest_out_2 = tmp_path / "guest_out_2.csv"
    run_cli(
        [
            "plan",
            "--host-roster",
            str(host_roster),
            "--guest-roster",
            str(guest_roster),
            "--state",
            str(state_path),
            "--event-index",
            "2",
            "--seed",
            "11",
            "--cohort-seed",
            "22",
            "--hosts-per-event",
            "0",
            "--guests-per-event",
            "4",
            "--guest-max-unique",
            "6",
            "--total-events",
            "2",
            "--host-out",
            str(tmp_path / "host_out_2.csv"),
            "--host-wait",
            str(tmp_path / "host_wait_2.csv"),
            "--guest-out",
            str(guest_out_2),
            "--guest-wait",
            str(tmp_path / "guest_wait_2.csv"),
        ]
    )
    assert "Unique Astro" in read_column(guest_out_2, "name")


def test_configurable_demographic_column_balances_targets(tmp_path: Path) -> None:
    host_roster = tmp_path / "hosts.csv"
    guest_roster = tmp_path / "guests_track.csv"
    state_path = tmp_path / "state.json"
    write_csv(host_roster, ["Name", "Discipline"], [["Host One", "biophysics"]])
    write_csv(
        guest_roster,
        ["Name", "Discipline", "Track", "Outcome"],
        [
            ["G1", "biophysics", "A", ""],
            ["G2", "astronomy", "A", ""],
            ["G3", "condensed matter experimental", "A", ""],
            ["G4", "condensed matter theory", "A", ""],
            ["G5", "biophysics", "A", ""],
            ["G6", "astronomy", "A", ""],
            ["G7", "condensed matter experimental", "B", ""],
            ["G8", "condensed matter theory", "B", ""],
        ],
    )

    run_cli(
        [
            "init",
            "--host-roster",
            str(host_roster),
            "--guest-roster",
            str(guest_roster),
            "--guest-demographic-column",
            "Track",
            "--state",
            str(state_path),
        ]
    )

    selected_tracks: Counter[str] = Counter()
    for event_index in (1, 2, 3, 4):
        guest_out = tmp_path / f"guest_out_{event_index}.csv"
        run_cli(
            [
                "plan",
                "--host-roster",
                str(host_roster),
                "--guest-roster",
                str(guest_roster),
                "--guest-demographic-column",
                "Track",
                "--state",
                str(state_path),
                "--event-index",
                str(event_index),
                "--seed",
                "2026",
                "--cohort-seed",
                "3030",
                "--hosts-per-event",
                "0",
                "--guests-per-event",
                "1",
                "--guest-max-unique",
                "8",
                "--total-events",
                "4",
                "--host-out",
                str(tmp_path / f"host_out_{event_index}.csv"),
                "--host-wait",
                str(tmp_path / f"host_wait_{event_index}.csv"),
                "--guest-out",
                str(guest_out),
                "--guest-wait",
                str(tmp_path / f"guest_wait_{event_index}.csv"),
            ]
        )
        selected = read_column(guest_out, "name")
        selected_track = read_column(guest_out, "track")
        selected_tracks.update(selected_track)

        set_guest_outcomes_csv(guest_roster, attended=selected, no_show=[])
        run_cli(
            [
                "record",
                "--state",
                str(state_path),
                "--event-index",
                str(event_index),
                "--guest-roster",
                str(guest_roster),
            ]
        )

    assert selected_tracks["A"] == 3
    assert selected_tracks["B"] == 1


def test_record_requires_guest_outcomes_for_selected(tmp_path: Path) -> None:
    host_roster = tmp_path / "hosts.csv"
    guest_roster = tmp_path / "guests.csv"
    state_path = tmp_path / "state.json"
    make_host_roster(host_roster)
    make_guest_roster(guest_roster)

    run_cli(
        [
            "init",
            "--host-roster",
            str(host_roster),
            "--guest-roster",
            str(guest_roster),
            "--state",
            str(state_path),
        ]
    )
    run_cli(
        [
            "plan",
            "--host-roster",
            str(host_roster),
            "--guest-roster",
            str(guest_roster),
            "--state",
            str(state_path),
            "--event-index",
            "1",
            "--seed",
            "10",
            "--cohort-seed",
            "20",
            "--hosts-per-event",
            "1",
            "--guests-per-event",
            "2",
            "--total-events",
            "4",
            "--host-out",
            str(tmp_path / "host_out.csv"),
            "--host-wait",
            str(tmp_path / "host_wait.csv"),
            "--guest-out",
            str(tmp_path / "guest_out.csv"),
            "--guest-wait",
            str(tmp_path / "guest_wait.csv"),
        ]
    )

    with pytest.raises(ValueError, match="Missing guest outcomes"):
        cli_main(
            [
                "record",
                "--state",
                str(state_path),
                "--event-index",
                "1",
                "--guest-roster",
                str(guest_roster),
            ]
        )


def test_xlsx_workbook_two_sheet_flow(tmp_path: Path) -> None:
    from openpyxl import Workbook, load_workbook

    workbook_path = tmp_path / "planner_inputs.xlsx"
    wb = Workbook()

    ws = wb.active
    ws.title = "host_roster"
    ws.append(["Name", "Discipline"])
    ws.append(["Host A", "biophysics"])
    ws.append(["Host B", "astronomy"])

    ws = wb.create_sheet("guest_roster")
    ws.append(["Name", "Discipline", "Sex", "Outcome"])
    ws.append(["Guest 1", "biophysics", "F", ""])
    ws.append(["Guest 2", "astronomy", "M", ""])
    ws.append(["Guest 3", "condensed matter experimental", "F", ""])
    ws.append(["Guest 4", "condensed matter theory", "M", ""])
    wb.save(workbook_path)

    state_path = tmp_path / "state.json"
    run_cli(
        [
            "init",
            "--host-roster",
            str(workbook_path),
            "--host-roster-sheet",
            "host_roster",
            "--guest-roster",
            str(workbook_path),
            "--guest-roster-sheet",
            "guest_roster",
            "--state",
            str(state_path),
        ]
    )

    guest_out = tmp_path / "guest_out.csv"
    run_cli(
        [
            "plan",
            "--host-roster",
            str(workbook_path),
            "--host-roster-sheet",
            "host_roster",
            "--guest-roster",
            str(workbook_path),
            "--guest-roster-sheet",
            "guest_roster",
            "--state",
            str(state_path),
            "--event-index",
            "1",
            "--seed",
            "77",
            "--cohort-seed",
            "88",
            "--hosts-per-event",
            "1",
            "--guests-per-event",
            "2",
            "--total-events",
            "4",
            "--host-out",
            str(tmp_path / "host_out.csv"),
            "--host-wait",
            str(tmp_path / "host_wait.csv"),
            "--guest-out",
            str(guest_out),
            "--guest-wait",
            str(tmp_path / "guest_wait.csv"),
        ]
    )

    selected_guests = read_column(guest_out, "name")
    assert len(selected_guests) == 2

    wb2 = load_workbook(workbook_path)
    ws2 = wb2["guest_roster"]
    header = [cell.value for cell in ws2[1]]
    name_idx = header.index("Name")
    outcome_idx = header.index("Outcome")
    for row in ws2.iter_rows(min_row=2):
        guest_name = (row[name_idx].value or "")
        row[outcome_idx].value = "attended" if guest_name in selected_guests else ""
    wb2.save(workbook_path)

    run_cli(
        [
            "record",
            "--state",
            str(state_path),
            "--event-index",
            "1",
            "--guest-roster",
            str(workbook_path),
            "--guest-roster-sheet",
            "guest_roster",
        ]
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    for guest_name in selected_guests:
        assert state["guests"][guest_name]["assigned_count"] == 1


def test_legacy_cli_still_works(tmp_path: Path) -> None:
    roster = tmp_path / "legacy_roster.csv"
    state_path = tmp_path / "legacy_state.json"
    write_csv(
        roster,
        ["Name", "Sex", "Discipline"],
        [
            ["Legacy A", "F", "biophysics"],
            ["Legacy B", "M", "astronomy"],
            ["Legacy C", "F", "condensed matter experimental"],
            ["Legacy D", "M", "condensed matter theory"],
            ["Legacy E", "F", "biophysics"],
        ],
    )

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "lunches.py", "init", "--roster", str(roster), "--state", str(state_path)],
        cwd=repo_root,
        check=True,
    )
    selected = tmp_path / "legacy_selected.csv"
    waitlist = tmp_path / "legacy_wait.csv"
    subprocess.run(
        [
            sys.executable,
            "lunches.py",
            "plan",
            "--roster",
            str(roster),
            "--state",
            str(state_path),
            "--lunch-index",
            "1",
            "--seed",
            "100",
            "--out",
            str(selected),
            "--wait",
            str(waitlist),
            "--seats",
            "3",
            "--max-unique",
            "4",
        ],
        cwd=repo_root,
        check=True,
    )
    assert selected.exists()
    assert waitlist.exists()

