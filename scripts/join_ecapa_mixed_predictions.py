#!/usr/bin/env python3
"""Join two-second mixed ground truth with ECAPA predictions."""

from pathlib import Path
import csv


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCORES_PATH = PROJECT_DIR / "results" / "ecapa_scores.csv"
SUMMARY_PATH = PROJECT_DIR / "results" / "ecapa_summary.csv"
GROUND_TRUTH_PATH = PROJECT_DIR / "results" / "mixed_2s_ground_truth.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "ecapa_mixed_2s_evaluation.csv"


with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as summary_file:
    summary_rows = list(csv.DictReader(summary_file))

two_second_summary = next(
    (row for row in summary_rows if float(row["window_seconds"]) == 2.0),
    None,
)
if two_second_summary is None:
    raise RuntimeError("No two-second threshold found in ecapa_summary.csv")

threshold = float(two_second_summary["exploratory_threshold"])

with SCORES_PATH.open("r", newline="", encoding="utf-8-sig") as scores_file:
    score_rows = list(csv.DictReader(scores_file))

scores_by_path = {
    row["window_path"]: row
    for row in score_rows
    if row["category"] == "mixed" and float(row["window_seconds"]) == 2.0
}

with GROUND_TRUTH_PATH.open("r", newline="", encoding="utf-8-sig") as truth_file:
    truth_rows = list(csv.DictReader(truth_file))

output_rows = []

for truth in truth_rows:
    score = scores_by_path.get(truth["window_path"])
    if score is None:
        raise RuntimeError(f"Missing ECAPA score for {truth['window_path']}")

    similarity = float(score["similarity"])
    prediction = "accept" if similarity >= threshold else "reject"
    target_speaking = truth["target_speaking"].strip().lower()

    if target_speaking == "yes":
        expected_decision = "accept"
        correct = prediction == expected_decision
    elif target_speaking == "no":
        expected_decision = "reject"
        correct = prediction == expected_decision
    elif target_speaking == "uncertain":
        expected_decision = "uncertain"
        correct = None
    else:
        expected_decision = "unlabeled"
        correct = None

    output_rows.append(
        {
            "window_path": truth["window_path"],
            "source_file": truth["source_file"],
            "start_seconds": truth["start_seconds"],
            "end_seconds": truth["end_seconds"],
            "target_speaking": target_speaking,
            "notes": truth["notes"],
            "similarity": score["similarity"],
            "threshold": f"{threshold:.2f}",
            "model_prediction": prediction,
            "expected_decision": expected_decision,
            "correct": "" if correct is None else str(correct).lower(),
            "inference_ms": score["inference_ms"],
        }
    )

fields = [
    "window_path",
    "source_file",
    "start_seconds",
    "end_seconds",
    "target_speaking",
    "notes",
    "similarity",
    "threshold",
    "model_prediction",
    "expected_decision",
    "correct",
    "inference_ms",
]

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(output_rows)

scored_rows = [row for row in output_rows if row["correct"]]
correct_rows = sum(row["correct"] == "true" for row in scored_rows)

print(f"Joined rows: {len(output_rows)}")
print(f"Scored rows: {len(scored_rows)}")
print(f"Uncertain/unscored rows: {len(output_rows) - len(scored_rows)}")
print(f"Correct decisions: {correct_rows}/{len(scored_rows)}")
print(f"Combined evaluation: {OUTPUT_PATH}")
