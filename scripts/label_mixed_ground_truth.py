#!/usr/bin/env python3
"""Interactively play and label mixed-audio ground-truth windows on macOS."""

from pathlib import Path
import csv
import subprocess


PROJECT_DIR = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = PROJECT_DIR / "results" / "mixed_2s_ground_truth.csv"


def save(rows: list[dict[str, str]], fields: list[str]) -> None:
    with GROUND_TRUTH_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


with GROUND_TRUTH_PATH.open("r", newline="", encoding="utf-8-sig") as input_file:
    reader = csv.DictReader(input_file)
    fields = list(reader.fieldnames or [])
    rows = list(reader)

unlabeled_indexes = [
    index
    for index, row in enumerate(rows)
    if not (row.get("target_speaking") or "").strip()
]

print(f"Unlabeled windows: {len(unlabeled_indexes)}")
print("y = target speaking, n = no target, u = uncertain, r = replay, q = quit")

for position, row_index in enumerate(unlabeled_indexes, start=1):
    row = rows[row_index]
    audio_path = PROJECT_DIR / row["window_path"]

    while True:
        print(
            f"\n[{position}/{len(unlabeled_indexes)}] "
            f"{row['source_file']} "
            f"{row['start_seconds']}-{row['end_seconds']}s"
        )
        subprocess.run(["afplay", str(audio_path)], check=True)
        answer = input("Label [y/n/u/r/q]: ").strip().lower()

        if answer == "r":
            continue
        if answer == "q":
            print("Stopped. Completed labels were already saved.")
            raise SystemExit(0)
        if answer not in {"y", "n", "u"}:
            print("Please enter y, n, u, r, or q.")
            continue

        row["target_speaking"] = {
            "y": "yes",
            "n": "no",
            "u": "uncertain",
        }[answer]
        save(rows, fields)
        break

print("\nAll mixed-window labels are complete and saved.")
