from __future__ import annotations

import argparse

from .pipeline import run_post_train_evaluation_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-train evaluation from a saved manifest")
    parser.add_argument("--manifest", required=True, help="Path to training_manifest.json")
    args = parser.parse_args()

    rows = run_post_train_evaluation_from_manifest(args.manifest)
    print(f"post_train_eval_manifest={args.manifest}")
    print(f"post_train_eval_rows={len(rows)}")


if __name__ == "__main__":
    main()
