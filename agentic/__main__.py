"""PulseFi agent commands."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "help"
    if command == "worker":
        sys.argv = ["agentic.worker", *args[1:]]
        from agentic.worker import main as run_worker

        return run_worker()
    if command == "dry-run":
        sys.argv = ["agentic.dry_run", *args[1:]]
        from agentic.dry_run import main as run_dry_run

        return run_dry_run()
    if command == "stream":
        sys.argv = ["agentic.stream_generator", *args[1:]]
        from agentic.stream_generator import main as run_stream

        return run_stream()
    print("Commands:")
    print("  python -m agentic worker   # real continuous LLM worker")
    print("  python -m agentic dry-run  # real API, synthetic BPM inputs")
    print("  python -m agentic stream   # generated live-prediction input only")
    print(
        "  streamlit run agentic/dashboard.py  # agent alerts and memories"
    )
    return 0 if command == "help" else 1


if __name__ == "__main__":
    raise SystemExit(main())
