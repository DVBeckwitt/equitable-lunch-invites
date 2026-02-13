# equitable-lunch-invites
i made this to plan lunches with a number of hosts and guests to keep it fair.
It invites from one workbook and one seed key.

## Equity rules considered

- Rotation fairness: fewer prior assignments are favored.
- Reliability weighting: no-shows reduce future priority.
- Demographic balancing: per-event deficits are corrected against series targets.
- Discipline coverage: each event tries to avoid discipline drop-out.
- Deterministic reproducibility: same inputs + seed-key gives the same plan.

## Input

One `.xlsx` file with:
- `host_roster`: `Name`, `Discipline`
- `guest_roster`: `Name`, `Discipline`, `Sex` (or your demographic column)

Use `templates/planner_templates.xlsx` as the format.
If `Discipline` is blank, missing entries are auto-distributed evenly across all disciplines.



## GUI (main.py)

Run:

```bash
python main.py
```

In the GUI:
- Set your workbook path, seed key, and output paths.
- Click `Start` for the first run.
- If `state.json` already contains a previous run, `Start` is disabled and `Next` is enabled.
- For later events, mark attendance on selected guest rows, click `Save attendance`, then click `Next`.
- The window uses a resizable split layout: controls on the left, results on the right.
- On smaller screens, the left control panel is scrollable and the results section is split into adjustable checklist/table panes.
- Use `Event filter` and `Status` (`all`, `selected`, `waitlist`) to view rows.
- Hosts are fixed to 1 per event.
- The results table shows guests only.
- You can edit attendance for previous events using `Event filter`.
- Use the attendance checkboxes for both selected and waitlist guest rows.
- Selected guest outcomes: `Attended`, `Can't Attend`, `No Show`.
- Waitlist guest outcomes: `Filled In`, `Can't Attend`, `No Show`.
- Checkbox edits are written to `lunch_plan.csv` immediately.
- `Reset All Data` deletes both `state.json` and `lunch_plan.csv` so you can restart cleanly.

Math specification:
- Math explanation (PDF): [docs/explanation.pdf](docs/explanation.pdf)

Writes:
- `data/lunch_plan.csv`
- `data/state.json`

Default demographic mode: `women_to_parity`.
- For `Sex`, women are upweighted to men for target math.
- Use `--demographic-mode proportional` to keep raw roster proportions.

## Next event flow

1. In `data/lunch_plan.csv`, fill `attendance` for selected guest rows (`attended`, `cant_attend`, or `no_show`) and optional waitlist fill-ins (`filled`).
2. Run `plan` again (omit `--event-index` to auto-advance).

The planner reads prior attendance from the same CSV before selecting the next event.
