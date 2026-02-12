# equitable-lunch-invites

Deterministic, fairness-aware event planning for selecting **N hosts** and **M guests**.

## Current workflow

You now need only:
1. `host_roster`
2. `guest_roster` (with an additional outcome column)

No separate host attendance/no-show files are required.
No separate guest-total input is required.

## Inputs

### Host roster (`.csv` or `.xlsx`)
Required columns:
- `Name`
- `Discipline`

### Guest roster (`.csv` or `.xlsx`)
Required columns:
- `Name`
- `Discipline`
- demographic column (default: `Sex`)
- outcome column (default: `Outcome`)

`Outcome` values:
- `attended`
- `no_show`
- blank (allowed for non-selected guests)

For selected guests in an event, `Outcome` must be filled before running `record`.

### Sheets (Excel)

CSV has no sheets.  
For one file with sheets, use `templates/planner_templates.xlsx`.

Template workbook sheets:
- `host_roster`
- `guest_roster`

## State behavior

State is tracked in `state.json`:
- `hosts.<name>.{assigned_count,no_show_count,cooldown}`
- `guests.<name>.{assigned_count,no_show_count,cooldown}`
- `_meta.event_history` keeps each planned event’s selected host/guest names

Host behavior:
- hosts are automatically recorded as attended during `record`
- host no-show input is not used

Guest behavior:
- outcomes are read from the guest roster `Outcome` column during `record`
- guest cooldown/no-show tracking remains active

## CLI

Install:

```bash
python -m pip install -e .
```

### 1) Initialize

```bash
equitable-invites init \
  --host-roster templates/planner_templates.xlsx --host-roster-sheet host_roster \
  --guest-roster templates/planner_templates.xlsx --guest-roster-sheet guest_roster \
  --state state.json
```

### 2) Plan

```bash
equitable-invites plan \
  --host-roster templates/planner_templates.xlsx --host-roster-sheet host_roster \
  --guest-roster templates/planner_templates.xlsx --guest-roster-sheet guest_roster \
  --state state.json \
  --event-index 1 \
  --seed 12345 \
  --cohort-seed 777 \
  --hosts-per-event 2 \
  --guests-per-event 5 \
  --total-events 5 \
  --host-out outputs/event1_hosts_selected.csv \
  --host-wait outputs/event1_hosts_waitlist.csv \
  --guest-out outputs/event1_guests_selected.csv \
  --guest-wait outputs/event1_guests_waitlist.csv
```

`--guest-max-unique` is optional.  
If omitted, it defaults to total guests in the guest roster.

### 3) Record outcomes

Update `Outcome` in `guest_roster` for the selected guests, then run:

```bash
equitable-invites record \
  --state state.json \
  --event-index 1 \
  --guest-roster templates/planner_templates.xlsx \
  --guest-roster-sheet guest_roster
```

## Data safety

Only templates under `templates/` should be tracked for CSV/XLSX.
Real data should stay untracked (for example under `data/`).
