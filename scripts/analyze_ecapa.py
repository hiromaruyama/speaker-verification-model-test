from pathlib import Path
import csv
from collections import defaultdict


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCORES_PATH = PROJECT_DIR / "results" / "ecapa_scores.csv"
SUMMARY_PATH = PROJECT_DIR / "results" / "ecapa_summary.csv"

with SCORES_PATH.open("r", newline="", encoding="utf-8") as scores_file:
    rows = list(csv.DictReader(scores_file))

if not rows:
    raise RuntimeError("No ECAPA scores were found")

groups = defaultdict(list)

for row in rows:
    window_seconds = float(row["window_seconds"])
    category = row["category"]
    similarity = float(row["similarity"])

    groups[(window_seconds, category)].append(similarity)


thresholds = [
    value / 100
    for value in range(0, 51)
]

print(f"Loaded {len(rows)} ECAPA scores\n")

summary_rows = []

for window_seconds in (1.0, 1.5, 2.0):
    clean_scores = groups[(window_seconds, "clean")]
    music_scores = groups[(window_seconds, "music_only")]
    mixed_scores = groups[(window_seconds, "mixed")]

    best_result = None

    for threshold in thresholds:
        clean_accept_rate = sum(
            score >= threshold for score in clean_scores
        ) / len(clean_scores)

        music_false_accept_rate = sum(
            score >= threshold for score in music_scores
        ) / len(music_scores)

        music_reject_rate = 1.0 - music_false_accept_rate

        balanced_accuracy = (
            clean_accept_rate + music_reject_rate
        ) / 2

        result = {
            "threshold": threshold,
            "clean_accept_rate": clean_accept_rate,
            "music_false_accept_rate": music_false_accept_rate,
            "balanced_accuracy": balanced_accuracy,
        }

        if (
            best_result is None
            or result["balanced_accuracy"]
            > best_result["balanced_accuracy"]
        ):
            best_result = result

    mixed_accept_rate = sum(
        score >= best_result["threshold"]
        for score in mixed_scores
    ) / len(mixed_scores)

    summary_rows.append(
        {
            "window_seconds": f"{window_seconds:g}",
            "exploratory_threshold": f"{best_result['threshold']:.2f}",
            "clean_window_count": len(clean_scores),
            "clean_accept_rate": f"{best_result['clean_accept_rate']:.6f}",
            "music_window_count": len(music_scores),
            "music_false_accept_rate": (
                f"{best_result['music_false_accept_rate']:.6f}"
            ),
            "balanced_accuracy": f"{best_result['balanced_accuracy']:.6f}",
            "mixed_window_count": len(mixed_scores),
            "mixed_accept_rate": f"{mixed_accept_rate:.6f}",
        }
    )

    print(f"Window: {window_seconds:g} seconds")
    print(f"  Exploratory threshold: {best_result['threshold']:.2f}")
    print(
        f"  Clean acceptance: "
        f"{best_result['clean_accept_rate']:.1%}"
    )
    print(
        f"  Music false acceptance: "
        f"{best_result['music_false_accept_rate']:.1%}"
    )
    print(
        f"  Balanced accuracy: "
        f"{best_result['balanced_accuracy']:.1%}"
    )
    print(
        f"  Mixed-window acceptance: "
        f"{mixed_accept_rate:.1%}"
    )
    print()


summary_fields = [
    "window_seconds",
    "exploratory_threshold",
    "clean_window_count",
    "clean_accept_rate",
    "music_window_count",
    "music_false_accept_rate",
    "balanced_accuracy",
    "mixed_window_count",
    "mixed_accept_rate",
]

with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as summary_file:
    writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Summary saved to: {SUMMARY_PATH}")
