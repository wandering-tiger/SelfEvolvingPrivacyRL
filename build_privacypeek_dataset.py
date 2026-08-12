"""Build PrivacyPeek defender training dataset."""

from __future__ import annotations

import argparse

from defender.dataset_builder import build_privacypeek_dataset, default_cases_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PrivacyPeek defender dataset")
    parser.add_argument(
        "--cases-dir",
        type=str,
        default=str(default_cases_dir()),
        help="PrivacyPeek data/cases directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/privacypeek",
        help="Output directory for train.json / val.json",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default=None,
        help="Comma-separated acquisition dimensions, e.g. A1,B1,B3",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dims = [d.strip() for d in args.dims.split(",")] if args.dims else None
    train_path, val_path = build_privacypeek_dataset(
        cases_dir=args.cases_dir,
        output_dir=args.output_dir,
        dims=dims,
        limit=args.limit,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Wrote train dataset: {train_path}")
    print(f"Wrote val dataset:   {val_path}")


if __name__ == "__main__":
    main()
