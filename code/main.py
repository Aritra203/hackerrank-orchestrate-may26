from __future__ import annotations

import argparse
from pathlib import Path

from agent import TriageAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HackerRank / Claude / Visa support triage agent")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("support_tickets/support_tickets.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("support_tickets/output.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = args.input if args.input.is_absolute() else repo_root / args.input
    output_csv = args.output if args.output.is_absolute() else repo_root / args.output
    corpus_root = repo_root / "data"

    agent = TriageAgent(repo_root=repo_root, corpus_root=corpus_root)
    agent.run(input_csv=input_csv, output_csv=output_csv)


if __name__ == "__main__":
    main()
