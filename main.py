#!/usr/bin/env python3
"""Command-line app: face -> product model -> voice -> display product.

Examples
    python main.py                                   interactive mode
    python main.py --face F.jpg --voice V.wav        one transaction
    python main.py --demo                            scripted authorized + unauthorized scenarios
    python main.py --check                           show which models are ready
    python main.py --mock ...                        use stand-ins for models that are not ready yet

Exit codes: 0 access granted, 1 access denied, 2 error (bad input / model problem).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

import system_integration as si  # noqa: E402

LOG_PATH = BASE_DIR / "logs" / "access_log.csv"
DEMO_FILE = BASE_DIR / "demo_scenarios.json"
RESULTS_PATH = BASE_DIR / "results" / "system_test_results.md"

_USE_COLOR = sys.stdout.isatty()
_COLORS = {"ok": "\033[32m", "fail": "\033[31m", "info": "\033[36m", "end": "\033[0m"}


def _paint(kind: str, text: str) -> str:
    return f"{_COLORS[kind]}{text}{_COLORS['end']}" if _USE_COLOR and kind in _COLORS else text


def make_reporter(delay: float = 0.0):
    marks = {"info": "->", "ok": "[OK]", "fail": "[X]"}

    def report(kind: str, text: str) -> None:
        print(_paint(kind, f"  {marks[kind]} {text}"), flush=True)
        if delay:
            time.sleep(delay)
    return report


def build_system(args, reporter) -> si.SecureRecommendationSystem:
    components = si.load_components(allow_mock=args.mock)
    if components.mocked:
        banner = (f"  [MOCK] Using stand-in(s) for: {', '.join(components.mocked)} "
                  "(real models not ready - results are NOT real predictions)")
        print(_paint("fail", banner) if not args.json else banner, file=sys.stderr if args.json else sys.stdout)
    return si.SecureRecommendationSystem(components.face_fn, components.voice_fn, components.product_fn, reporter=reporter)


def print_result(result: si.TransactionResult) -> None:
    kind = "ok" if result.granted else "fail"
    print()
    print(_paint(kind, "  " + result.message))
    print()


def append_log(face, voice, result: si.TransactionResult) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new = not LOG_PATH.exists()
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            if new:
                fh.write("timestamp,face_file,voice_file,status,identity\n")
            fh.write(f"{datetime.now().isoformat(timespec='seconds')},{Path(str(face)).name},"
                     f"{Path(str(voice)).name},{result.status},{result.identity or ''}\n")
    except OSError:
        pass  # logging must never break the app


def exit_code(result: si.TransactionResult) -> int:
    return 0 if result.granted else (2 if result.status == si.ERROR else 1)


# ----------------------------------------------------------------------------- modes
def ask_path(prompt: str):
    try:
        value = input(prompt).strip()
    except EOFError:
        return ""
    return value


def run_interactive(args) -> int:
    print("=" * 62)
    print(" Secure Product Recommendation System")
    print(" Face check -> prediction -> voice check -> result")
    print("=" * 62)
    system = build_system(args, make_reporter())
    last = 0
    while True:
        print("\nNew transaction (press Enter with no path to quit)")
        face = ask_path("  Path to your face image : ")
        if not face:
            return last
        result = system.run_transaction(face, lambda: ask_path("  Path to your voice recording (e.g. 'Confirm transaction') : "))
        print_result(result)
        if not args.no_log:
            append_log(face, "interactive", result)
        last = exit_code(result)
        if input("Run another transaction? [y/N] ").strip().lower() != "y":
            return last


def run_single(args) -> int:
    reporter = None if args.json else make_reporter()
    system = build_system(args, reporter)
    result = system.run_transaction(args.face, args.voice)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_result(result)
    if not args.no_log:
        append_log(args.face, args.voice, result)
    return exit_code(result)


def run_demo(args) -> int:
    try:
        scenarios = json.loads(DEMO_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        print(f"Cannot read {DEMO_FILE.name}: {err}")
        return 2
    print("=" * 62)
    print(" SYSTEM DEMONSTRATION - authorized and unauthorized attempts")
    print("=" * 62)
    components = si.load_components(allow_mock=args.mock)
    if components.mocked:
        print(_paint("fail", f"[MOCK] stand-ins in use for: {', '.join(components.mocked)}"))
    rows, all_pass, skipped, skipped_names = [], True, 0, []
    for number, sc in enumerate(scenarios, 1):
        print(f"\nScenario {number}: {sc['name']}")
        print(f"  face : {sc['face']}\n  voice: {sc['voice']}")
        missing = [k for k in ("face", "voice") if not (BASE_DIR / sc[k]).exists()]
        if sc.get("optional") and missing:
            skipped += 1
            skipped_names.append((number, sc))
            print(_paint("info", f"  SKIPPED - optional scenario, file not found. {sc.get('note', '')}"))
            continue
        system = si.SecureRecommendationSystem(components.face_fn, components.voice_fn, components.product_fn,
                                               reporter=make_reporter(args.delay))
        result = system.run_transaction(BASE_DIR / sc["face"], BASE_DIR / sc["voice"])
        print_result(result)
        passed = result.status == sc["expected"]
        all_pass &= passed
        rows.append((number, sc, result, passed))
        time.sleep(args.delay)
    note = f" ({skipped} optional scenario(s) skipped)" if skipped else ""
    print(f"Summary: {sum(r[3] for r in rows)}/{len(rows)} scenarios behaved as expected.{note}")
    if not args.no_results_file:
        write_results(rows, components, skipped_names)
    return 0 if all_pass else 1


def write_results(rows, components: si.Components, skipped_names=()) -> None:
    mode = ("REAL models" if not components.mocked
            else f"STAND-INS used for: {', '.join(components.mocked)} (others real)")
    lines = ["# System Testing Results", "",
             f"Run: {datetime.now().isoformat(timespec='seconds')}  |  Mode: {mode}", "",
             "| # | Scenario | Face file | Face result | Voice file | Voice result | Expected | Actual | Result |",
             "|---|----------|-----------|-------------|------------|--------------|----------|--------|--------|"]

    def cell(v, ran: bool) -> str:
        if v is None:
            return "not run" if not ran else "-"
        who = v.identity or "rejected"
        if v.similarity is not None:  # face model: decided by cosine similarity
            return f"{who} (similarity {v.similarity:.3f})"
        return f"{who} (confidence {v.confidence:.3f})" if v.confidence is not None else who

    for number, sc, result, passed in rows:
        lines.append(f"| {number} | {sc['name']} | {Path(sc['face']).name} | {cell(result.face, False)} | "
                     f"{Path(sc['voice']).name} | {cell(result.voice, False)} | "
                     f"{sc['expected']} | {result.status} | {'PASS' if passed else 'FAIL'} |")
    for number, sc in skipped_names:
        lines.append(f"| {number} | {sc['name']} | {Path(sc['face']).name} | - | {Path(sc['voice']).name} | - | "
                     f"{sc['expected']} | SKIPPED | add `{sc['face']}` to run |")
    try:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Results table written to {RESULTS_PATH.relative_to(BASE_DIR)}")
    except OSError as err:
        print(f"Could not write results file: {err}")


def run_check() -> int:
    print("Model readiness:")
    status = si.component_status()
    for name, value in status.items():
        print(f"  {name:8s} {value}")
    return 0 if all(v == "READY" for v in status.values()) else 1


# ----------------------------------------------------------------------------- entry point
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Face + voice authenticated product recommendation (CLI).")
    p.add_argument("--face", help="path to a face image (.jpg/.jpeg/.png/.heic)")
    p.add_argument("--voice", help="path to a voice recording (.wav/.mp3/.m4a/.flac/.ogg)")
    p.add_argument("--demo", action="store_true", help="run the scripted authorized + unauthorized scenarios")
    p.add_argument("--check", action="store_true", help="show which of the three models are ready")
    p.add_argument("--mock", action="store_true", help="use stand-ins for any model that is not ready yet")
    p.add_argument("--json", action="store_true", help="print the result as JSON (single transaction)")
    p.add_argument("--no-log", action="store_true", help="do not append to logs/access_log.csv")
    p.add_argument("--delay", type=float, default=0.6, help="pause between demo steps in seconds (default 0.6)")
    p.add_argument("--no-results-file", action="store_true", help="demo: do not write results/system_test_results.md")
    args = p.parse_args(argv)
    if bool(args.face) != bool(args.voice):
        p.error("--face and --voice must be given together")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.check:
            return run_check()
        if args.demo:
            return run_demo(args)
        if args.face and args.voice:
            return run_single(args)
        return run_interactive(args)
    except si.IntegrationError as err:
        print(f"Setup problem: {err}")
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
