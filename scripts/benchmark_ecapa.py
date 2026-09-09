import csv
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier


# Resolve paths relative to the project folder.
PROJECT_DIR = Path(__file__).resolve().parents[1]

NORMALIZED_DIR = PROJECT_DIR / "recordings" / "normalized"
WINDOWS_DIR = PROJECT_DIR / "recordings" / "windows"
MANIFEST_PATH = PROJECT_DIR / "recordings" / "windows.csv"

MODEL_DIR = PROJECT_DIR / "models" / "ecapa-tdnn"
RESULTS_DIR = PROJECT_DIR / "results"
SCORES_PATH = RESULTS_DIR / "ecapa_scores.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load ECAPA-TDNN locally on the CPU.
load_started = time.perf_counter()

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR),
    run_opts={"device": "cpu"},
)

load_seconds = time.perf_counter() - load_started

print(f"ECAPA loaded in {load_seconds:.3f} seconds")
print(f"Normalized recordings: {NORMALIZED_DIR}")
print(f"Window manifest: {MANIFEST_PATH}")
print(f"Results will be saved to: {SCORES_PATH}")

# Find all six enrollment recordings.
enrollment_files = sorted(NORMALIZED_DIR.glob("enroll_*.wav"))

if len(enrollment_files) != 6:
    raise RuntimeError(
        f"Expected 6 enrollment recordings, found {len(enrollment_files)}"
    )

print("\nCreating enrollment profile...")

enrollment_embeddings = []

with torch.inference_mode():
    for enrollment_file in enrollment_files:
        started = time.perf_counter()

        waveform = model.load_audio(str(enrollment_file))
        embedding = model.encode_batch(waveform.unsqueeze(0))
        embedding = F.normalize(embedding, p=2, dim=-1)

        elapsed = time.perf_counter() - started
        enrollment_embeddings.append(embedding)

        print(
            f"  {enrollment_file.name}: "
            f"shape={tuple(embedding.shape)}, "
            f"time={elapsed:.3f}s"
        )

# Average all six voice representations into one profile.
enrollment_profile = torch.stack(enrollment_embeddings).mean(dim=0)
enrollment_profile = F.normalize(enrollment_profile, p=2, dim=-1)

print("\nEnrollment profile created")
print(f"Profile shape: {tuple(enrollment_profile.shape)}")
print(f"Profile norm: {enrollment_profile.norm().item():.4f}")

# Read the window manifest.
with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as manifest_file:
    manifest_rows = list(csv.DictReader(manifest_file))

if not manifest_rows:
    raise RuntimeError("The window manifest is empty")

print(f"\nScoring {len(manifest_rows)} test windows...")

result_fields = list(manifest_rows[0].keys()) + [
    "similarity",
    "inference_ms",
]

benchmark_started = time.perf_counter()

with SCORES_PATH.open("w", newline="", encoding="utf-8") as scores_file:
    writer = csv.DictWriter(scores_file, fieldnames=result_fields)
    writer.writeheader()

    with torch.inference_mode():
        for index, row in enumerate(manifest_rows, start=1):
            window_path = PROJECT_DIR / row["window_path"]

            # Audio loading is kept outside the inference timer.
            waveform = model.load_audio(str(window_path))

            inference_started = time.perf_counter()

            test_embedding = model.encode_batch(waveform.unsqueeze(0))
            test_embedding = F.normalize(test_embedding, p=2, dim=-1)

            similarity = F.cosine_similarity(
                enrollment_profile,
                test_embedding,
                dim=-1,
            ).item()

            inference_ms = (
                time.perf_counter() - inference_started
            ) * 1000

            result = dict(row)
            result["similarity"] = f"{similarity:.6f}"
            result["inference_ms"] = f"{inference_ms:.3f}"

            writer.writerow(result)

            if index % 100 == 0 or index == len(manifest_rows):
                print(f"  Scored {index}/{len(manifest_rows)}")


benchmark_seconds = time.perf_counter() - benchmark_started

print("\nScoring complete")
print(f"Total benchmark time: {benchmark_seconds:.2f} seconds")
print(f"Scores saved to: {SCORES_PATH}")