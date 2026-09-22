"""PulseFi agent commands."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "help"
    if command == "harness":
        from agentic.harness import main as run_harness

        return run_harness()
    if command == "worker":
        sys.argv = ["agentic.worker", *args[1:]]
        from agentic.worker import main as run_worker

        return run_worker()
    print("Commands:")
    print("  python -m agentic harness  # offline plumbing test; no API")
    print("  python -m agentic worker   # real continuous LLM worker")
    return 0 if command == "help" else 1


if __name__ == "__main__":
    raise SystemExit(main())
