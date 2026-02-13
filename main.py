from __future__ import annotations

import csv
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tkinter import END, BooleanVar, Canvas, StringVar, Tk, filedialog, messagebox, ttk

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from equitable_lunch_invites.cli import (  # noqa: E402
    DEFAULT_GUEST_ROSTER_SHEET,
    DEFAULT_HOST_ROSTER_SHEET,
)
from equitable_lunch_invites.cli import main as cli_main  # noqa: E402
from equitable_lunch_invites.io import PLAN_CSV_HEADER  # noqa: E402
from equitable_lunch_invites.models import (  # noqa: E402
    DEFAULT_DEMOGRAPHIC_MODE,
    DEFAULT_GUEST_DEMOGRAPHIC_COLUMN,
    DEMOGRAPHIC_MODE_PROPORTIONAL,
    DEMOGRAPHIC_MODE_WOMEN_TO_PARITY,
)


class PlannerGui:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Equitable Lunch Invites")
        self.root.geometry("1320x860")
        self.root.minsize(980, 620)

        self.workbook_var = StringVar(value="templates/planner_templates.xlsx")
        self.state_var = StringVar(value="data/state.json")
        self.plan_csv_var = StringVar(value="data/lunch_plan.csv")
        self.seed_key_var = StringVar(value="42")
        self.host_sheet_var = StringVar(value=DEFAULT_HOST_ROSTER_SHEET)
        self.guest_sheet_var = StringVar(value=DEFAULT_GUEST_ROSTER_SHEET)
        self.demographic_column_var = StringVar(value=DEFAULT_GUEST_DEMOGRAPHIC_COLUMN)
        self.demographic_mode_var = StringVar(value=DEFAULT_DEMOGRAPHIC_MODE)

        self.event_index_var = StringVar(value="")
        self.guests_per_event_var = StringVar(value="5")
        self.total_events_var = StringVar(value="5")
        self.guest_max_unique_var = StringVar(value="")
        self.waitlist_size_var = StringVar(value="8")
        self.event_filter_var = StringVar(value="all")
        self.status_filter_var = StringVar(value="all")

        self.plan_rows: list[dict[str, str]] = []
        self.tree_to_row_index: dict[str, int] = {}
        self.attendance_var_pairs: list[tuple[BooleanVar, BooleanVar, BooleanVar]] = []

        self._configure_style()
        self._build_layout()
        self.refresh_table()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("Root.TFrame", background="#f3f5f8")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure(
            "Title.TLabel",
            font=("Segoe UI Semibold", 12),
            foreground="#1f2937",
            background="#ffffff",
        )
        style.configure(
            "Muted.TLabel",
            font=("Segoe UI", 9),
            foreground="#64748b",
            background="#ffffff",
        )
        style.configure("Primary.TButton", padding=(10, 7))
        style.configure("Accent.TButton", padding=(10, 7))
        style.configure("Danger.TButton", padding=(10, 7))
        style.configure("TLabelframe", padding=8)
        style.configure("TLabelframe.Label", font=("Segoe UI Semibold", 10))

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=10)
        root_frame.pack(fill="both", expand=True)

        outer_pane = ttk.Panedwindow(root_frame, orient="horizontal")
        outer_pane.pack(fill="both", expand=True)

        left_panel = ttk.Frame(outer_pane, style="Panel.TFrame", padding=(8, 8))
        right_panel = ttk.Frame(outer_pane, style="Panel.TFrame", padding=(8, 8))
        outer_pane.add(left_panel, weight=0)
        outer_pane.add(right_panel, weight=1)

        # Scrollable left sidebar: controls remain reachable on short screens.
        self.sidebar_canvas = Canvas(left_panel, highlightthickness=0, bd=0, background="#ffffff")
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar_scroll = ttk.Scrollbar(left_panel, orient="vertical", command=self.sidebar_canvas.yview)
        sidebar_scroll.pack(side="right", fill="y")
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)

        self.sidebar_inner = ttk.Frame(self.sidebar_canvas, style="Panel.TFrame", padding=(4, 4))
        self.sidebar_window_id = self.sidebar_canvas.create_window((0, 0), window=self.sidebar_inner, anchor="nw")
        self.sidebar_inner.bind("<Configure>", self._on_sidebar_inner_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)

        ttk.Label(self.sidebar_inner, text="Planner Control Center", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            self.sidebar_inner,
            text="Start first run once, then use Next for each subsequent event.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 8))

        config_frame = ttk.LabelFrame(self.sidebar_inner, text="Configuration")
        config_frame.pack(fill="x")
        self._add_row(config_frame, 0, "Workbook (.xlsx)", self.workbook_var, browse="xlsx")
        self._add_row(config_frame, 1, "State file (.json)", self.state_var, browse="json")
        self._add_row(config_frame, 2, "Plan CSV", self.plan_csv_var, browse="csv")
        self._add_row(config_frame, 3, "Seed key (0-1000)", self.seed_key_var)
        self._add_row(config_frame, 4, "Host sheet", self.host_sheet_var)
        self._add_row(config_frame, 5, "Guest sheet", self.guest_sheet_var)
        self._add_row(config_frame, 6, "Demographic column", self.demographic_column_var)
        ttk.Label(config_frame, text="Demographic mode").grid(row=7, column=0, sticky="w", padx=(0, 8), pady=4)
        mode_combo = ttk.Combobox(
            config_frame,
            textvariable=self.demographic_mode_var,
            state="readonly",
            values=[DEMOGRAPHIC_MODE_WOMEN_TO_PARITY, DEMOGRAPHIC_MODE_PROPORTIONAL],
            width=20,
        )
        mode_combo.grid(row=7, column=1, sticky="w", pady=4)

        plan_frame = ttk.LabelFrame(self.sidebar_inner, text="Planning")
        plan_frame.pack(fill="x", pady=(8, 0))
        self._add_row(plan_frame, 0, "Event index (optional)", self.event_index_var)
        ttk.Label(plan_frame, text="Hosts per event is fixed to 1", style="Muted.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=(0, 8), pady=4
        )
        self._add_row(plan_frame, 2, "Guests per event", self.guests_per_event_var)
        self._add_row(plan_frame, 3, "Total events", self.total_events_var)
        self._add_row(plan_frame, 4, "Guest max unique (optional)", self.guest_max_unique_var)
        self._add_row(plan_frame, 5, "Waitlist size", self.waitlist_size_var)

        action_frame = ttk.LabelFrame(self.sidebar_inner, text="Actions")
        action_frame.pack(fill="x", pady=(8, 0))
        self.start_button = ttk.Button(
            action_frame,
            text="Start",
            command=self.run_start,
            style="Primary.TButton",
            width=16,
        )
        self.start_button.grid(row=0, column=0, padx=(0, 6), pady=(2, 6), sticky="ew")
        self.next_button = ttk.Button(
            action_frame,
            text="Next",
            command=self.run_next,
            style="Primary.TButton",
            width=16,
        )
        self.next_button.grid(row=0, column=1, padx=(0, 0), pady=(2, 6), sticky="ew")
        ttk.Button(
            action_frame,
            text="Refresh Results",
            command=self.refresh_table,
            style="Accent.TButton",
            width=16,
        ).grid(row=1, column=0, padx=(0, 6), pady=(0, 2), sticky="ew")
        ttk.Button(
            action_frame,
            text="Reset All Data",
            command=self.reset_all_data,
            style="Danger.TButton",
            width=16,
        ).grid(row=1, column=1, padx=(0, 0), pady=(0, 2), sticky="ew")
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)

        log_frame = ttk.LabelFrame(self.sidebar_inner, text="Log")
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log = ttk.Treeview(log_frame, columns=("message",), show="headings", height=8)
        self.log.heading("message", text="message")
        self.log.column("message", width=380, anchor="w")
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        log_scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=log_scroll.set)

        results_header = ttk.Frame(right_panel, style="Panel.TFrame")
        results_header.pack(fill="x")
        ttk.Label(results_header, text="Guest Results", style="Title.TLabel").pack(side="left")
        ttk.Button(results_header, text="Save attendance", command=self.save_plan_csv, width=16).pack(side="right")

        filter_bar = ttk.Frame(right_panel, style="Panel.TFrame")
        filter_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(filter_bar, text="Event filter").pack(side="left")
        self.filter_combo = ttk.Combobox(
            filter_bar,
            textvariable=self.event_filter_var,
            state="readonly",
            values=["all"],
            width=8,
        )
        self.filter_combo.pack(side="left", padx=(6, 0))
        self.filter_combo.bind("<<ComboboxSelected>>", lambda _: self.render_table())
        ttk.Label(filter_bar, text="Status").pack(side="left", padx=(12, 0))
        self.status_filter_combo = ttk.Combobox(
            filter_bar,
            textvariable=self.status_filter_var,
            state="readonly",
            values=["all", "selected", "waitlist"],
            width=10,
        )
        self.status_filter_combo.pack(side="left", padx=(6, 0))
        self.status_filter_combo.bind("<<ComboboxSelected>>", lambda _: self.render_table())
        self.summary_var = StringVar(value="No rows loaded.")
        ttk.Label(filter_bar, textvariable=self.summary_var, style="Muted.TLabel").pack(side="left", padx=(14, 0))

        results_split = ttk.Panedwindow(right_panel, orient="vertical")
        results_split.pack(fill="both", expand=True, pady=(8, 0))

        checklist_frame = ttk.LabelFrame(results_split, text="Attendance Checklist")
        results_split.add(checklist_frame, weight=0)
        self.attendance_canvas = Canvas(checklist_frame, height=190, highlightthickness=0, bd=0, background="#ffffff")
        self.attendance_canvas.pack(side="left", fill="both", expand=True)
        attendance_scroll = ttk.Scrollbar(checklist_frame, orient="vertical", command=self.attendance_canvas.yview)
        attendance_scroll.pack(side="right", fill="y")
        self.attendance_canvas.configure(yscrollcommand=attendance_scroll.set)
        self.attendance_inner = ttk.Frame(self.attendance_canvas, style="Panel.TFrame")
        self.attendance_window_id = self.attendance_canvas.create_window((0, 0), window=self.attendance_inner, anchor="nw")
        self.attendance_inner.bind("<Configure>", self._on_attendance_inner_configure)
        self.attendance_canvas.bind("<Configure>", self._on_attendance_canvas_configure)

        table_frame = ttk.LabelFrame(results_split, text="Plan Rows")
        results_split.add(table_frame, weight=1)
        tree_container = ttk.Frame(table_frame, style="Panel.TFrame")
        tree_container.pack(fill="both", expand=True)
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        columns = PLAN_CSV_HEADER
        self.tree = ttk.Treeview(tree_container, columns=columns, show="headings", selectmode="extended")
        for col in columns:
            self.tree.heading(col, text=col)
            if col == "event_index":
                width = 76
            elif col in {"role", "status"}:
                width = 90
            elif col in {"name", "discipline"}:
                width = 220
            elif col == "attendance":
                width = 120
            else:
                width = 100
            self.tree.column(col, width=width, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(tree_container, orient="horizontal", command=self.tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        ttk.Label(
            right_panel,
            text=(
                "Attendance checkboxes auto-save to lunch_plan.csv. "
                "Waitlist rows can be marked as Filled In / Can't Attend / No Show."
            ),
            style="Muted.TLabel",
        ).pack(fill="x", pady=(6, 0))

        # Initial pane placement tuned for smaller laptop displays.
        self.root.after(10, lambda: self._set_sash_positions(outer_pane, results_split))

        self.update_action_buttons()

    def _set_sash_positions(self, outer_pane: ttk.Panedwindow, results_split: ttk.Panedwindow) -> None:
        try:
            outer_pane.sashpos(0, 420)
            results_split.sashpos(0, 230)
        except Exception:  # noqa: BLE001
            pass

    def _on_sidebar_inner_configure(self, _: object) -> None:
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, event: object) -> None:
        width = int(getattr(event, "width", 0) or 0)
        if width > 0:
            self.sidebar_canvas.itemconfigure(self.sidebar_window_id, width=width)

    def _add_row(
        self,
        parent: ttk.Frame | ttk.LabelFrame,
        row: int,
        label: str,
        variable: StringVar,
        browse: str | None = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=30)
        entry.grid(row=row, column=1, sticky="we", pady=4)
        parent.grid_columnconfigure(1, weight=1)
        if browse:
            ttk.Button(parent, text="Browse", command=lambda: self.browse_file(variable, browse), width=9).grid(
                row=row, column=2, padx=(8, 0), pady=4
            )

    def browse_file(self, variable: StringVar, kind: str) -> None:
        if kind == "xlsx":
            file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        elif kind == "json":
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
        else:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
        if file_path:
            variable.set(file_path)
            self.update_action_buttons()

    def run_next(self) -> None:
        args = [
            "plan",
            "--inputs",
            self.workbook_var.get().strip(),
            "--state",
            self.state_var.get().strip(),
            "--plan-csv",
            self.plan_csv_var.get().strip(),
            "--hosts-per-event",
            "1",
            "--guests-per-event",
            self.guests_per_event_var.get().strip(),
            "--total-events",
            self.total_events_var.get().strip(),
            "--waitlist-size",
            self.waitlist_size_var.get().strip(),
            "--host-roster-sheet",
            self.host_sheet_var.get().strip(),
            "--guest-roster-sheet",
            self.guest_sheet_var.get().strip(),
            "--guest-demographic-column",
            self.demographic_column_var.get().strip(),
            "--demographic-mode",
            self.demographic_mode_var.get().strip(),
        ]
        event_index = self.event_index_var.get().strip()
        guest_max_unique = self.guest_max_unique_var.get().strip()
        if event_index:
            args.extend(["--event-index", event_index])
        if guest_max_unique:
            args.extend(["--guest-max-unique", guest_max_unique])
        self._run_command(args, "next")

    def run_start(self) -> None:
        args = [
            "start",
            "--inputs",
            self.workbook_var.get().strip(),
            "--seed-key",
            self.seed_key_var.get().strip(),
            "--state",
            self.state_var.get().strip(),
            "--plan-csv",
            self.plan_csv_var.get().strip(),
            "--hosts-per-event",
            "1",
            "--guests-per-event",
            self.guests_per_event_var.get().strip(),
            "--total-events",
            self.total_events_var.get().strip(),
            "--waitlist-size",
            self.waitlist_size_var.get().strip(),
            "--host-roster-sheet",
            self.host_sheet_var.get().strip(),
            "--guest-roster-sheet",
            self.guest_sheet_var.get().strip(),
            "--guest-demographic-column",
            self.demographic_column_var.get().strip(),
            "--demographic-mode",
            self.demographic_mode_var.get().strip(),
        ]
        event_index = self.event_index_var.get().strip()
        guest_max_unique = self.guest_max_unique_var.get().strip()
        if event_index:
            args.extend(["--event-index", event_index])
        if guest_max_unique:
            args.extend(["--guest-max-unique", guest_max_unique])
        self._run_command(args, "start")

    def _run_command(self, args: list[str], label: str) -> None:
        if not self.workbook_var.get().strip():
            messagebox.showerror("Missing input", "Workbook path is required.")
            return

        log_buffer = io.StringIO()
        try:
            with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
                exit_code = cli_main(args)
            if exit_code != 0:
                raise RuntimeError(f"Command returned non-zero exit code: {exit_code}")
        except SystemExit as exc:
            self._append_log(f"{label} failed: {exc}")
            messagebox.showerror("Command failed", f"{label} failed: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            details = log_buffer.getvalue().strip()
            suffix = f"\n\n{details}" if details else ""
            self._append_log(f"{label} failed: {exc}")
            messagebox.showerror("Command failed", f"{label} failed: {exc}{suffix}")
            return

        output = log_buffer.getvalue().strip()
        if output:
            first_line = output.splitlines()[0]
            self._append_log(f"{label}: {first_line}")
        else:
            self._append_log(f"{label}: completed")

        self.refresh_table()

    def _append_log(self, message: str) -> None:
        self.log.insert("", 0, values=(message,))
        # Keep log compact.
        entries = self.log.get_children()
        if len(entries) > 300:
            for item in entries[300:]:
                self.log.delete(item)

    def _has_previous_run(self) -> bool:
        state_path = Path(self.state_var.get().strip())
        if not state_path.exists():
            return False

        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        if not isinstance(payload, dict):
            return False
        meta = payload.get("_meta", {})
        if not isinstance(meta, dict):
            return False

        last_planned = meta.get("last_planned_event", 0)
        try:
            if int(last_planned or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass

        event_history = meta.get("event_history", {})
        return isinstance(event_history, dict) and bool(event_history)

    def update_action_buttons(self) -> None:
        previous_run_detected = self._has_previous_run()
        if previous_run_detected:
            self.start_button.state(["disabled"])
            self.next_button.state(["!disabled"])
        else:
            self.start_button.state(["!disabled"])
            self.next_button.state(["disabled"])

    def reset_all_data(self) -> None:
        state_path = Path(self.state_var.get().strip())
        plan_path = Path(self.plan_csv_var.get().strip())
        message = (
            "This will delete planner state and plan CSV files:\n\n"
            f"- {state_path}\n"
            f"- {plan_path}\n\n"
            "Continue?"
        )
        if not messagebox.askyesno("Reset all data", message):
            return

        deleted: list[str] = []
        missing: list[str] = []
        errors: list[str] = []
        for path in (state_path, plan_path):
            if not path.exists():
                missing.append(str(path))
                continue
            try:
                path.unlink()
                deleted.append(str(path))
            except OSError as exc:
                errors.append(f"{path}: {exc}")

        self.plan_rows = []
        self.event_filter_var.set("all")
        self.status_filter_var.set("all")
        self.filter_combo.configure(values=["all"])
        self.render_table()
        self.update_action_buttons()

        summary_lines = []
        if deleted:
            summary_lines.append("Deleted:")
            summary_lines.extend(f"- {item}" for item in deleted)
        if missing:
            summary_lines.append("Not found:")
            summary_lines.extend(f"- {item}" for item in missing)
        if errors:
            summary_lines.append("Errors:")
            summary_lines.extend(f"- {item}" for item in errors)

        summary = "\n".join(summary_lines) if summary_lines else "Nothing changed."
        self._append_log("Reset all data executed.")
        if errors:
            messagebox.showwarning("Reset completed with errors", summary)
        else:
            messagebox.showinfo("Reset completed", summary)

    def refresh_table(self) -> None:
        path = Path(self.plan_csv_var.get().strip())
        if not path.exists():
            self.plan_rows = []
            self.event_filter_var.set("all")
            self.status_filter_var.set("all")
            self.filter_combo.configure(values=["all"])
            self.render_table()
            self.update_action_buttons()
            self._append_log(f"Plan CSV not found at {path}")
            return

        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            self.plan_rows = [{k: (v or "") for k, v in row.items()} for row in reader]

        events = sorted(
            {
                row.get("event_index", "")
                for row in self.plan_rows
                if row.get("role", "") == "guest" and row.get("event_index", "")
            },
            key=self._event_sort_key,
        )
        filter_values = ["all"] + events
        current_filter = self.event_filter_var.get() or "all"
        if current_filter not in filter_values:
            current_filter = "all"
        self.event_filter_var.set(current_filter)
        self.filter_combo.configure(values=filter_values)

        self.render_table()
        self.update_action_buttons()
        self._append_log(f"Loaded {len(self.plan_rows)} rows from {path}")

    @staticmethod
    def _event_sort_key(value: str) -> tuple[int, str]:
        token = (value or "").strip()
        if token.isdigit():
            return (0, f"{int(token):08d}")
        return (1, token)

    def render_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree_to_row_index.clear()

        selected_event = self.event_filter_var.get().strip() or "all"
        selected_status = self.status_filter_var.get().strip().lower() or "all"
        visible_count = 0
        selected_count = 0
        waitlist_count = 0
        for index, row in enumerate(self.plan_rows):
            if row.get("role", "") != "guest":
                continue
            if selected_event != "all" and row.get("event_index", "") != selected_event:
                continue

            status = (row.get("status", "") or "").strip().lower()
            if status == "selected":
                selected_count += 1
            if status == "waitlist":
                waitlist_count += 1
            if selected_status != "all" and status != selected_status:
                continue

            item_id = self.tree.insert(
                "",
                END,
                values=[row.get(column, "") for column in PLAN_CSV_HEADER],
            )
            self.tree_to_row_index[str(item_id)] = index
            visible_count += 1

        summary = (
            f"Showing {visible_count} guest rows "
            f"(selected: {selected_count}, waitlist: {waitlist_count})"
        )
        if selected_event != "all":
            summary += f" for event {selected_event}"
        if selected_status != "all":
            summary += f" [{selected_status}]"
        self.summary_var.set(summary)
        self.render_attendance_checkboxes()

    def _on_attendance_inner_configure(self, _: object) -> None:
        self.attendance_canvas.configure(scrollregion=self.attendance_canvas.bbox("all"))

    def _on_attendance_canvas_configure(self, event: object) -> None:
        width = int(getattr(event, "width", 0) or 0)
        if width > 0:
            self.attendance_canvas.itemconfigure(self.attendance_window_id, width=width)

    def render_attendance_checkboxes(self) -> None:
        for child in self.attendance_inner.winfo_children():
            child.destroy()
        self.attendance_var_pairs.clear()

        selected_event = self.event_filter_var.get().strip() or "all"
        selected_status = self.status_filter_var.get().strip().lower() or "all"
        editable_rows: list[tuple[int, dict[str, str]]] = []
        for index, row in enumerate(self.plan_rows):
            if row.get("role") != "guest":
                continue
            row_status = (row.get("status") or "").strip().lower()
            if row_status not in {"selected", "waitlist"}:
                continue
            if selected_event != "all" and row.get("event_index", "") != selected_event:
                continue
            if selected_status != "all" and row_status != selected_status:
                continue
            editable_rows.append((index, row))

        if not editable_rows:
            ttk.Label(
                self.attendance_inner,
                text="No guest rows available for attendance editing in this view.",
            ).grid(row=0, column=0, sticky="w", padx=4, pady=4)
            return

        ttk.Label(self.attendance_inner, text="Guest").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(self.attendance_inner, text="Attended/Filled").grid(row=0, column=1, sticky="w", padx=8, pady=2)
        ttk.Label(self.attendance_inner, text="Can't Attend").grid(row=0, column=2, sticky="w", padx=8, pady=2)
        ttk.Label(self.attendance_inner, text="No Show").grid(row=0, column=3, sticky="w", padx=8, pady=2)

        for line, (row_index, row) in enumerate(editable_rows, start=1):
            event_index = (row.get("event_index") or "").strip()
            name = (row.get("name") or "").strip()
            row_status = (row.get("status") or "").strip().lower()
            status_tag = "waitlist" if row_status == "waitlist" else "selected"
            label = (
                f"Event {event_index} [{status_tag}]: {name}"
                if selected_event == "all"
                else f"[{status_tag}] {name}"
            )
            ttk.Label(self.attendance_inner, text=label).grid(row=line, column=0, sticky="w", padx=4, pady=2)

            current = (row.get("attendance") or "").strip().lower()
            primary_key = "filled" if row_status == "waitlist" else "attended"
            primary_var = BooleanVar(value=(current == primary_key or (row_status == "waitlist" and current == "attended")))
            cant_attend_var = BooleanVar(value=current == "cant_attend")
            no_show_var = BooleanVar(value=current == "no_show")
            self.attendance_var_pairs.append((primary_var, cant_attend_var, no_show_var))
            primary_label = "Filled In" if row_status == "waitlist" else "Attended"

            ttk.Checkbutton(
                self.attendance_inner,
                text=primary_label,
                variable=primary_var,
                command=lambda i=row_index, s=row_status, p=primary_var, c=cant_attend_var, n=no_show_var: self.on_attendance_checkbox_changed(
                    i, s, p, c, n, "primary"
                ),
            ).grid(row=line, column=1, sticky="w", padx=8, pady=2)
            ttk.Checkbutton(
                self.attendance_inner,
                text="Can't Attend",
                variable=cant_attend_var,
                command=lambda i=row_index, s=row_status, p=primary_var, c=cant_attend_var, n=no_show_var: self.on_attendance_checkbox_changed(
                    i, s, p, c, n, "cant_attend"
                ),
            ).grid(row=line, column=2, sticky="w", padx=8, pady=2)
            ttk.Checkbutton(
                self.attendance_inner,
                text="No Show",
                variable=no_show_var,
                command=lambda i=row_index, s=row_status, p=primary_var, c=cant_attend_var, n=no_show_var: self.on_attendance_checkbox_changed(
                    i, s, p, c, n, "no_show"
                ),
            ).grid(row=line, column=3, sticky="w", padx=8, pady=2)

    def on_attendance_checkbox_changed(
        self,
        row_index: int,
        row_status: str,
        primary_var: BooleanVar,
        cant_attend_var: BooleanVar,
        no_show_var: BooleanVar,
        changed: str,
    ) -> None:
        if row_index < 0 or row_index >= len(self.plan_rows):
            return

        row = self.plan_rows[row_index]
        normalized_status = (row.get("status") or "").strip().lower()
        if row.get("role") != "guest" or normalized_status not in {"selected", "waitlist"}:
            return

        if changed == "primary" and primary_var.get():
            cant_attend_var.set(False)
            no_show_var.set(False)
        if changed == "cant_attend" and cant_attend_var.get():
            primary_var.set(False)
            no_show_var.set(False)
        if changed == "no_show" and no_show_var.get():
            primary_var.set(False)
            cant_attend_var.set(False)

        primary_value = "filled" if row_status == "waitlist" else "attended"
        if primary_var.get():
            row["attendance"] = primary_value
        elif cant_attend_var.get():
            row["attendance"] = "cant_attend"
        elif no_show_var.get():
            row["attendance"] = "no_show"
        else:
            row["attendance"] = ""

        for item_id, mapped_index in self.tree_to_row_index.items():
            if mapped_index != row_index:
                continue
            self.tree.item(item_id, values=[row.get(column, "") for column in PLAN_CSV_HEADER])
            break

        self.save_plan_csv(notify=False)

    def save_plan_csv(self, notify: bool = True) -> None:
        path = Path(self.plan_csv_var.get().strip())
        if not self.plan_rows:
            if notify:
                messagebox.showinfo("No data", "No plan rows loaded.")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PLAN_CSV_HEADER)
            writer.writeheader()
            writer.writerows(self.plan_rows)

        if notify:
            self._append_log(f"Saved {len(self.plan_rows)} row(s) to {path}")
            messagebox.showinfo("Saved", f"Saved attendance updates to {path}")


def main() -> None:
    root = Tk()
    PlannerGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
