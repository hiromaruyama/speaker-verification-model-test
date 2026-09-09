#!/usr/bin/env python3
"""Create model-independent ground truth for two-second mixed windows."""

from pathlib import Path
import csv


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCORES_PATH = PROJECT_DIR / "results" / "ecapa_scores.csv"
REVIEW_PATH = PROJECT_DIR / "results" / "ecapa_mixed_2s_review.csv"
GROUND_TRUTH_PATH = PROJECT_DIR / "results" / "mixed_2s_ground_truth.csv"


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        rows = csv.DictReader(input_file)
        return {
            row["window_path"]: {
                "target_speaking": (row.get("target_speaking") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
            for row in rows
        }


# Preserve any labels already entered in the ground-truth file. The earlier
# rejected-window review supplies initial labels when no ground-truth label exists.
saved_labels = read_labels(GROUND_TRUTH_PATH)
review_labels = read_labels(REVIEW_PATH)

with SCORES_PATH.open("r", newline="", encoding="utf-8-sig") as scores_file:
    score_rows = list(csv.DictReader(scores_file))

ground_truth_rows = []

for row in score_rows:
    if row["category"] != "mixed" or float(row["window_seconds"]) != 2.0:
        continue

    labels = saved_labels.get(row["window_path"])
    if labels is None:
        labels = review_labels.get(
            row["window_path"],
            {"target_speaking": "", "notes": ""},
        )

    ground_truth_rows.append(
        {
            "window_path": row["window_path"],
            "source_file": row["source_file"],
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "target_speaking": labels["target_speaking"],
            "notes": labels["notes"],
        }
    )

ground_truth_rows.sort(
    key=lambda row: (row["source_file"], float(row["start_seconds"]))
)

fields = [
    "window_path",
    "source_file",
    "start_seconds",
    "end_seconds",
    "target_speaking",
    "notes",
]

with GROUND_TRUTH_PATH.open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(ground_truth_rows)

labeled = sum(
    bool(row["target_speaking"].strip()) for row in ground_truth_rows
)

print(f"Two-second mixed windows: {len(ground_truth_rows)}")
print(f"Labels preserved or copied: {labeled}")
print(f"Labels remaining: {len(ground_truth_rows) - labeled}")
print(f"Ground truth: {GROUND_TRUTH_PATH}")
