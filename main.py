"""IIL — Institutional Intelligence Layer
Main CLI interface for the SFSU Academic Policy RAG engine.

Usage:
  python main.py --query "I'm transferring with 65 units..."          # Anonymous query
  python main.py --query "Am I on track?" --student-id STU-2024-0001  # Personalized
  python main.py --interactive                                          # REPL mode
  python main.py --interactive --student-id STU-2024-0001             # Personalized REPL
  python main.py --demo                                                 # Run README demo query
  python main.py --list-students                                        # Show all student IDs
  python main.py --eval                                                 # Run benchmark
  python main.py --eval --limit 5                                      # Quick 5-query eval
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


DEMO_QUERY = (
    "I'm transferring with 65 units. Can I still graduate in 2 years if I switch "
    "my major to Computer Science? I have a 3.2 GPA and completed Calc 1 and 2."
)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   INSTITUTIONAL INTELLIGENCE LAYER (IIL)                     ║
║   SFSU Academic Policy RAG Engine                            ║
║   94% Retrieval Accuracy | 1M+ Data Points Indexed           ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_student_header(student_id: str) -> None:
    """Print a header showing which student record is loaded."""
    try:
        from pipelines.student_store import StudentStore
        from rich.console import Console
        from rich.panel import Panel

        store = StudentStore()
        record = store.get(student_id)
        if record:
            console = Console()
            console.print(
                Panel(
                    f"[bold]Student:[/bold] {record.name}  |  "
                    f"[bold]ID:[/bold] {student_id}\n"
                    f"[bold]Major:[/bold] {record.major}  |  "
                    f"[bold]GPA:[/bold] {record.cumulative_gpa}  |  "
                    f"[bold]Standing:[/bold] {record.academic_standing}  |  "
                    f"[bold]Units:[/bold] {record.total_units}",
                    title="[bold blue]🎓 STUDENT RECORD LOADED[/bold blue]",
                    border_style="blue",
                )
            )
    except Exception:
        print(f"[Student ID: {student_id}]")


def cmd_list_students(args) -> None:
    """List all available student IDs."""
    from pipelines.student_store import StudentStore

    store = StudentStore()
    ids = store.list_ids()
    print(f"\nAvailable student records ({len(ids)} total):\n")
    for sid in ids:
        record = store.get(sid)
        if record:
            aid_status = record.financial_aid.get("status", "Unknown")
            print(
                f"  {sid}  |  {record.name:<20}  |  "
                f"Major: {record.major:<25}  |  "
                f"GPA: {record.cumulative_gpa}  |  "
                f"Standing: {record.academic_standing}  |  "
                f"Aid: {aid_status}"
            )
    print()


def print_response(response, query: str = "") -> None:
    """Pretty-print a FormattedResponse."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich.rule import Rule

        console = Console()

        if query:
            console.print(f"\n[bold cyan]QUERY:[/bold cyan] {query}")

        console.print(Rule(style="dim"))
        console.print(
            Panel(
                response.raw_answer,
                title="[bold green]IIL RESPONSE[/bold green]",
                border_style="green",
            )
        )

        if response.citations:
            console.print("\n[bold yellow]📚 CITATIONS[/bold yellow]")
            for c in response.citations:
                console.print(f"  • {c}")

        console.print(
            f"\n[dim]⚡ {response.processing_time_ms:.0f}ms | "
            f"Contexts: {response.num_contexts_used} | "
            f"Grounded: {'✅' if response.grounded else '❌'}[/dim]"
        )

    except ImportError:
        # Fallback without rich
        if query:
            print(f"\nQUERY: {query}")
        print("\n" + "─" * 60)
        print(response.to_display_text())
        print(f"\n⚡ {response.processing_time_ms:.0f}ms | Contexts: {response.num_contexts_used}")


def load_engine():
    """Load the IIL query engine (requires ingest.py to have been run first)."""
    from pipelines.indexer import load_existing_index
    from rag.query_engine import IILQueryEngine

    try:
        index = load_existing_index()
        return IILQueryEngine(index)
    except Exception as e:
        print(f"\n❌ Failed to load index: {e}")
        print("  Have you run the ingestion pipeline? → python ingest.py")
        sys.exit(1)


def cmd_query(args) -> None:
    """Run a single query, optionally personalized to a student."""
    student_id = getattr(args, "student_id", None)
    engine = load_engine()
    if student_id:
        print_student_header(student_id)
    response = engine.query(args.query, student_id=student_id, verbose=args.verbose)
    print_response(response, query=args.query)


def cmd_demo(args) -> None:
    """Run the README demo query."""
    print("\n📋 Running README demo query...")
    engine = load_engine()
    response = engine.query(DEMO_QUERY, verbose=True)
    print_response(response, query=DEMO_QUERY)


def cmd_interactive(args) -> None:
    """Interactive REPL mode, optionally locked to a specific student."""
    engine = load_engine()
    student_id = getattr(args, "student_id", None)

    print(BANNER)

    if student_id:
        print_student_header(student_id)
        print("Advising session locked to student record above.")
        print("All responses will be personalized to their transcript.\n")
    else:
        print("Anonymous mode — no student record loaded.")
        print("Use --student-id STU-XXXX-XXXX to enable personalized advising.\n")

    print("Type your question and press Enter. Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            prompt = f"📝 [{student_id}] " if student_id else "📝 "
            query = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not query:
            continue

        response = engine.query(query, student_id=student_id, verbose=args.verbose)
        print_response(response)
        print()


def _import_from_file(file_path: str, data: dict) -> dict | None:
    """
    Import a student record from a JSON or CSV file.
    Returns the record dict on success, None on failure.
    """
    import csv
    import json
    from pathlib import Path

    p = Path(file_path)
    if not p.exists():
        print(f"❌ File not found: {file_path}")
        return None

    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        # Accept either a bare record dict or a {"records": {...}} wrapper
        if "records" in raw:
            records = raw["records"]
            if len(records) == 1:
                sid, record = next(iter(records.items()))
            else:
                print(f"Found {len(records)} records in file. Importing all.")
                added = []
                for sid, record in records.items():
                    if sid in data["records"]:
                        print(f"  ⚠️  {sid} already exists — skipped")
                    else:
                        data["records"][sid] = record
                        added.append(sid)
                        print(f"  ✅ Imported {sid} — {record.get('name', '')}")
                return {"_multi": True, "added": added}
        else:
            sid = raw.get("student_id")
            if not sid:
                print("❌ JSON record missing 'student_id' field.")
                return None
            record = raw
        if sid in data["records"]:
            print(f"⚠️  Student ID '{sid}' already exists. Aborting.")
            return None
        return record

    elif p.suffix.lower() == ".csv":
        # CSV format: one row per student (flat fields).
        # Courses are NOT included in CSV — they must be added via --add-student
        # or by editing the JSON directly.
        #
        # Required columns: student_id, name, declared_major, cumulative_gpa,
        #   units_transferred, units_completed_at_sfsu
        # Optional columns: email, admission_type, admit_term, catalog_year,
        #   college, academic_standing, enrollment_status, major_gpa,
        #   transfer_institution, igetc_certified, aid_status, aid_types,
        #   remaining_aid_years, sap_note, advisor_notes
        records_added = []
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("student_id", "").strip()
                if not sid:
                    print(f"  ⚠️  Row missing student_id — skipped: {row}")
                    continue
                if sid in data["records"]:
                    print(f"  ⚠️  {sid} already exists — skipped")
                    continue
                aid_types_raw = row.get("aid_types", "")
                aid_types = [t.strip() for t in aid_types_raw.split(",") if t.strip()]
                aid = {
                    "status": row.get("aid_status", "Unknown"),
                    "type": aid_types,
                    "remaining_aid_years": float(row.get("remaining_aid_years", 0) or 0),
                }
                if row.get("sap_note", "").strip():
                    aid["sap_note"] = row["sap_note"].strip()

                units_t = int(row.get("units_transferred", 0) or 0)
                units_s = int(row.get("units_completed_at_sfsu", 0) or 0)
                record = {
                    "student_id":               sid,
                    "name":                     row.get("name", "").strip(),
                    "email":                    row.get("email", "").strip(),
                    "admission_type":           row.get("admission_type", "transfer").strip(),
                    "admit_term":               row.get("admit_term", "").strip(),
                    "catalog_year":             row.get("catalog_year", "").strip(),
                    "declared_major":           row.get("declared_major", "").strip(),
                    "college":                  row.get("college", "").strip(),
                    "academic_standing":        row.get("academic_standing", "Good Standing").strip(),
                    "enrollment_status":        row.get("enrollment_status", "Full-Time").strip(),
                    "cumulative_gpa":           float(row.get("cumulative_gpa", 0) or 0),
                    "major_gpa":                float(row.get("major_gpa", 0) or 0),
                    "units_transferred":        units_t,
                    "units_completed_at_sfsu":  units_s,
                    "total_units_attempted":    units_t + units_s,
                    "financial_aid":            aid,
                    "transfer_institution":     row.get("transfer_institution", "").strip(),
                    "igetc_certified":          row.get("igetc_certified", "no").strip().lower() == "yes",
                    "transfer_courses_completed": [],
                    "sfsu_courses_completed":   [],
                    "courses_in_progress":      [],
                    "advisor_notes":            row.get("advisor_notes", "").strip(),
                }
                data["records"][sid] = record
                records_added.append(sid)
                print(f"  ✅ Imported {sid} — {record['name']}")

        if not records_added:
            print("No new records imported.")
            return None
        print(f"\nNote: CSV import does not include course history.")
        print(f"Add courses via: python main.py --add-student  (interactive, update existing ID)")
        return {"_multi": True, "added": records_added}

    else:
        print(f"❌ Unsupported file type: {p.suffix}. Use .json or .csv")
        return None


def cmd_add_student(args) -> None:
    """Interactively collect a student's transcript and write to student_records.json."""
    import json
    from config import settings

    records_path = settings.data_dir / "synthetic" / "student_records.json"
    data = json.loads(records_path.read_text(encoding="utf-8"))

    # --- File import mode ---
    from_file = getattr(args, "from_file", None)
    if from_file:
        result = _import_from_file(from_file, data)
        if result is None:
            return
        if result.get("_multi"):
            records_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"\n✅ {len(result['added'])} record(s) saved to student_records.json")
            return
        sid = result["student_id"]
        data["records"][sid] = result
        records_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\n✅ Student record saved: {sid} — {result.get('name', '')}")
        print(f"   Use: python main.py --interactive --student-id {sid}")
        return

    print("\n=== ADD STUDENT RECORD ===\n")

    def prompt(label, default=None):
        suffix = f" [{default}]" if default is not None else ""
        val = input(f"{label}{suffix}: ").strip()
        return val if val else default

    def prompt_float(label, default=0.0):
        while True:
            val = input(f"{label} [{default}]: ").strip()
            if not val:
                return default
            try:
                return float(val)
            except ValueError:
                print("  Enter a number (e.g. 3.2)")

    def prompt_int(label, default=0):
        while True:
            val = input(f"{label} [{default}]: ").strip()
            if not val:
                return default
            try:
                return int(val)
            except ValueError:
                print("  Enter a whole number")

    # --- Identity ---
    existing_ids = list(data["records"].keys())
    next_num = len(existing_ids) + 1
    default_id = f"STU-{2024}-{next_num:04d}"
    student_id = prompt("Student ID", default_id)

    if student_id in data["records"]:
        print(f"\n⚠️  Student ID '{student_id}' already exists. Aborting.")
        return

    name            = prompt("Full name")
    email           = prompt("Email", f"{name.split()[0].lower()}@sfsu.edu" if name else "")
    admission_type  = prompt("Admission type (transfer/freshman)", "transfer")
    admit_term      = prompt("Admit term (e.g. Fall 2024)", "Fall 2024")
    catalog_year    = prompt("Catalog year (e.g. 2024-25)", "2024-25")
    major           = prompt("Declared major")
    college         = prompt("College (e.g. Science & Engineering)")
    standing        = prompt("Academic standing", "Good Standing")
    enrollment      = prompt("Enrollment status (Full-Time/Part-Time)", "Full-Time")
    cum_gpa         = prompt_float("Cumulative GPA")
    major_gpa       = prompt_float("Major GPA")
    units_transfer  = prompt_int("Units transferred")
    units_sfsu      = prompt_int("Units completed at SFSU")
    transfer_inst   = prompt("Transfer institution (leave blank if freshman)", "")
    igetc           = prompt("IGETC certified? (yes/no)", "no").lower() == "yes"
    advisor_notes   = prompt("Advisor notes (optional)", "")

    # --- Financial Aid ---
    print("\n-- Financial Aid --")
    aid_status      = prompt("Aid status (Active/Inactive/SAP Warning)", "Active")
    aid_types_raw   = prompt("Aid types (comma-separated, e.g. Cal Grant A, Pell Grant)", "")
    aid_types       = [t.strip() for t in aid_types_raw.split(",") if t.strip()]
    aid_years       = prompt_float("Remaining aid years", 2.0)
    sap_note        = prompt("SAP note (leave blank if none)", "")

    # --- Courses ---
    def collect_courses(label, include_grade=True, include_sfsu_equiv=False):
        courses = []
        print(f"\n-- {label} --")
        print("Enter one course per line. Leave course code blank to finish.")
        if include_sfsu_equiv:
            print("Format: course code | SFSU equivalent | grade | units")
        elif include_grade:
            print("Format: course code | title | grade | units | term")
        else:
            print("Format: course code | title | term")

        while True:
            raw = input("  > ").strip()
            if not raw:
                break
            parts = [p.strip() for p in raw.split("|")]
            if include_sfsu_equiv:
                # transfer course: code | sfsu_equiv | grade | units
                c = {
                    "course":        parts[0] if len(parts) > 0 else "",
                    "sfsu_equivalent": parts[1] if len(parts) > 1 else "",
                    "grade":         parts[2] if len(parts) > 2 else "",
                    "units":         int(parts[3]) if len(parts) > 3 else 3,
                }
            elif include_grade:
                # sfsu completed: code | title | grade | units | term
                c = {
                    "course": parts[0] if len(parts) > 0 else "",
                    "title":  parts[1] if len(parts) > 1 else "",
                    "grade":  parts[2] if len(parts) > 2 else "",
                    "units":  int(parts[3]) if len(parts) > 3 else 3,
                    "term":   parts[4] if len(parts) > 4 else "",
                }
            else:
                # in-progress: code | title | term
                c = {
                    "course": parts[0] if len(parts) > 0 else "",
                    "title":  parts[1] if len(parts) > 1 else "",
                    "term":   parts[2] if len(parts) > 2 else "",
                }
            courses.append(c)
        return courses

    transfer_courses  = collect_courses("Transfer Courses Completed", include_sfsu_equiv=True)
    sfsu_courses      = collect_courses("SFSU Courses Completed", include_grade=True)
    in_progress       = collect_courses("Courses In Progress", include_grade=False)

    # --- Build record ---
    aid = {
        "status": aid_status,
        "type": aid_types,
        "remaining_aid_years": aid_years,
    }
    if sap_note:
        aid["sap_note"] = sap_note

    record = {
        "student_id":               student_id,
        "name":                     name,
        "email":                    email,
        "admission_type":           admission_type,
        "admit_term":               admit_term,
        "catalog_year":             catalog_year,
        "declared_major":           major,
        "college":                  college,
        "academic_standing":        standing,
        "enrollment_status":        enrollment,
        "cumulative_gpa":           cum_gpa,
        "major_gpa":                major_gpa,
        "units_transferred":        units_transfer,
        "units_completed_at_sfsu":  units_sfsu,
        "total_units_attempted":    units_transfer + units_sfsu,
        "financial_aid":            aid,
        "transfer_institution":     transfer_inst,
        "igetc_certified":          igetc,
        "transfer_courses_completed": transfer_courses,
        "sfsu_courses_completed":   sfsu_courses,
        "courses_in_progress":      in_progress,
        "advisor_notes":            advisor_notes,
    }

    data["records"][student_id] = record
    records_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"\n✅ Student record saved: {student_id} — {name}")
    print(f"   Use: python main.py --interactive --student-id {student_id}")


def cmd_eval(args) -> None:
    """Run the benchmark evaluation."""
    from evals.benchmark import run_benchmark
    run_benchmark(
        category=args.category,
        limit=args.limit,
        verbose=args.verbose,
    )


def cmd_ragas(args) -> None:
    """Run Ragas evaluation."""
    from evals.ragas_eval import run_ragas_evaluation
    run_ragas_evaluation(limit=args.limit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IIL — Institutional Intelligence Layer for SFSU Academic Advising",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo
  python main.py --query "What is the CS major GPA requirement?"
  python main.py --interactive
  python main.py --eval --limit 5
  python main.py --ragas --limit 5
        """,
    )

    sub = parser.add_subparsers(dest="cmd")

    # Single query
    q_parser = sub.add_parser("query", help="Run a single query")
    q_parser.add_argument("query", help="The student's question")
    q_parser.add_argument("--verbose", action="store_true")

    # Demo
    demo_p = sub.add_parser("demo", help="Run the README demo query")
    demo_p.add_argument("--verbose", action="store_true")

    # Interactive
    int_p = sub.add_parser("interactive", help="Interactive REPL mode")
    int_p.add_argument("--verbose", action="store_true")

    # Eval
    eval_p = sub.add_parser("eval", help="Run benchmark evaluation")
    eval_p.add_argument("--category", help="Filter by category", default=None)
    eval_p.add_argument("--limit", type=int, help="Limit queries", default=None)
    eval_p.add_argument("--verbose", action="store_true")

    # Ragas
    ragas_p = sub.add_parser("ragas", help="Run Ragas evaluation")
    ragas_p.add_argument("--limit", type=int, default=None)

    # Legacy --flags mode for simpler invocation
    parser.add_argument("--query", help="Single query mode")
    parser.add_argument("--demo", action="store_true", help="Run demo query")
    parser.add_argument("--interactive", action="store_true", help="REPL mode")
    parser.add_argument("--eval", action="store_true", help="Run benchmark")
    parser.add_argument("--ragas", action="store_true", help="Run Ragas eval")
    parser.add_argument("--list-students", action="store_true", help="List all student IDs")
    parser.add_argument("--add-student", action="store_true", help="Interactively add a new student record")
    parser.add_argument("--from-file", dest="from_file", default=None, help="Import student record(s) from a .json or .csv file")
    parser.add_argument(
        "--student-id",
        dest="student_id",
        default=None,
        help="Student ID (e.g. STU-2024-0001) to load a transcript and personalize responses",
    )
    parser.add_argument("--category", help="Eval category filter", default=None)
    parser.add_argument("--limit", type=int, help="Eval query limit", default=None)
    parser.add_argument("--verbose", action="store_true", help="Verbose pipeline output")

    args = parser.parse_args()

    # Handle subcommands
    if args.cmd == "query":
        cmd_query(args)
    elif args.cmd == "demo":
        cmd_demo(args)
    elif args.cmd == "interactive":
        cmd_interactive(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "ragas":
        cmd_ragas(args)
    # Handle legacy --flags
    elif getattr(args, "add_student", False) or getattr(args, "from_file", None):
        cmd_add_student(args)
    elif getattr(args, "list_students", False):
        cmd_list_students(args)
    elif args.query:
        cmd_query(args)
    elif args.demo:
        cmd_demo(args)
    elif args.interactive:
        cmd_interactive(args)
    elif args.eval:
        cmd_eval(args)
    elif args.ragas:
        cmd_ragas(args)
    else:
        # Default: show demo
        print(BANNER)
        print("No command specified. Showing demo...\n")
        args.verbose = False
        cmd_demo(args)


if __name__ == "__main__":
    main()
