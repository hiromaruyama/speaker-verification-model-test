# Speaker-verification-model-test:
Interned at Palkie Talkie; Evaluated speaker verification model called ECAPA-TDNN for integration.

## Context: 
Over the summer, I interned for PalkieTalkie, an AI-powered English tutoring app. The biggest challenge I faced was filtering out unwanted speech, background noise during AI-live conversations because AI was sometimes oversensitive and detected nearby speech that was not from an actual user. 


To address this, I proposed incorporating a speaker verification model and tested two architectures: ECAPA-TDNN and CAM++. I researched and tested the models across different environments, changing the background music and speech from music with lyrics to people talking, and varying the volume on a scale, while testing with buffer zones of 1, 1.5, and 2 seconds using the same 6 voice embeddings (1492 audio windows), while fine-tuning the threshold of rejection. The results came out to be 93.86% for average of clean speech acceptance rate and music only rejection rate, with 86.8% accuracy for mixed audio with background noise. 

## Takeaway: 
However, one of my main takeaways from this experience was that speaker verification introduces an important tradeoff. The system first needs a voice embedding created from an enrolled speaker’s audio. It also requires a sufficiently long audio segment, approximately one second at least in my testing, to determine whether the speech belongs to that speaker or not.

This was determinental for an app designed to be natural, real-time conversation. 


This experience taught me that selecting an AI model is not only about its accuracy. It is also about understanding how priorities of accuracy, latency, error rate, and goals could differ, and ultimately is about the user experience.


## Model used
This repository evaluates the pretrained [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) speaker-verification model on short audio windows. It measures how well a speaker profile built from enrollment recordings distinguishes clean target speech from music-only audio and explores performance when speech is mixed with music.

The repository contains the benchmark scripts and result tables. Model checkpoints, Python environments, caches, and evaluation recordings are intentionally excluded. The model is downloaded from its original source when the benchmark is first run.

## Benchmark outline

1. Split normalized test recordings into non-overlapping 1, 1.5, and 2-second windows.
2. Build one enrollment profile by averaging embeddings from six enrollment recordings.
3. Calculate cosine similarity between the enrollment profile and every test window.
4. Select an exploratory threshold for each window duration using clean speech and music-only examples.
5. Manually label 2-second mixed-audio windows and compare the model decisions with those labels.

recordings/normalized/enroll_03_en.wav
recordings/normalized/enroll_01_ja.wav
recordings/normalized/enroll_02_ja.wav
recordings/normalized/enroll_03_ja.wav


Test recordings are assigned to categories from their filenames:

- `clean_*.wav` — clean target-speaker audio; expected to be accepted
- `*_music_only_*.wav` — music without the target speaker; expected to be rejected
- `*_mixed_*.wav` — target speech and music; evaluated with manual labels

## Run the benchmark

Run commands from the repository root.

### 1. Split test recordings

```bash
python scripts/split_windows.py
```

This creates:

- `recordings/windows/` — generated WAV windows
- `recordings/windows.csv` — the window manifest

Alternative input and output paths can be supplied with `--input`, `--output`, and `--manifest`.

### 2. Score the windows with ECAPA-TDNN

```bash
python scripts/benchmark_ecapa.py
```

On the first run, SpeechBrain downloads `speechbrain/spkrec-ecapa-voxceleb` into `models/ecapa-tdnn/`. The script runs on CPU, averages the six normalized enrollment embeddings into one speaker profile, and writes per-window cosine similarities and inference times to:

```text
results/ecapa_scores.csv
```

### 3. Calculate exploratory thresholds

```bash
python scripts/analyze_ecapa.py
```

For each window duration, this searches thresholds from 0.00 through 0.50 in increments of 0.01. It selects the threshold with the highest balanced accuracy across clean target speech and music-only audio, then writes:

```text
results/ecapa_summary.csv
```

### 4. Create and label mixed-audio ground truth

```bash
python scripts/create_mixed_ground_truth.py
python scripts/label_mixed_ground_truth.py
```

The first command creates `results/mixed_2s_ground_truth.csv` for 2-second mixed windows while preserving existing labels. The labeling command plays each unlabeled window using macOS `afplay` and accepts:

- `y` — target speaker is speaking
- `n` — target speaker is not speaking
- `u` — uncertain
- `r` — replay the window
- `q` — save progress and quit

Each completed label is saved immediately.

### 5. Compare predictions with human labels

```bash
python scripts/join_ecapa_mixed_predictions.py
```

This combines the 2-second mixed-audio labels, ECAPA scores, and selected 2-second threshold. It writes model decisions and correctness values to:

```text
results/ecapa_mixed_2s_evaluation.csv
```

Windows labeled `uncertain` or left unlabeled are included in the output but excluded from the reported correctness count.

## Existing exploratory results

The included summary reports the following dataset-specific results:

| Window | Threshold | Clean acceptance | Music false acceptance | Balanced accuracy | Mixed acceptance |
|---:|---:|---:|---:|---:|---:|
| 1.0 s | 0.22 | 93.68% | 0.00% | 96.84% | 58.94% |
| 1.5 s | 0.14 | 100.00% | 0.00% | 100.00% | 83.33% |
| 2.0 s | 0.15 | 100.00% | 0.00% | 100.00% | 86.82% |

These values are exploratory measurements on this small evaluation set. The thresholds were selected and evaluated using the same clean and music-only samples, so they should not be interpreted as unbiased test-set performance or production-ready thresholds. Use separate development and test speakers, impostor trials, and independent validation data for a rigorous speaker-verification evaluation.

## Model attribution

The ECAPA-TDNN checkpoint used by this project is provided by SpeechBrain and trained on VoxCeleb data. Review the upstream model card, code license, and dataset terms before redistribution or commercial use:

- [SpeechBrain ECAPA-TDNN model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [SpeechBrain](https://github.com/speechbrain/speechbrain)



