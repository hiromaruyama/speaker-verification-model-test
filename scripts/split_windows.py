#!/usr/bin/env python3
"""Split normalized test audio into fixed, non-overlapping PCM WAV windows."""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path


WINDOW_SECONDS = (1.0, 1.5, 2.0)


def classify(stem: str) -> tuple[str, str]:
    if stem.startswith("clean_"):
        return "clean", "accept"
    if "_mixed_" in stem:
        return "mixed", "accept_when_target_speaking"
    if "_music_only_" in stem:
        return "music_only", "reject"
    raise ValueError(f"Unrecognized test filename: {stem}")


def window_tag(seconds: float) -> str:
    return f"{seconds:g}s".replace(".", "p")


def split_file(source: Path, output_root: Path, writer: csv.DictWriter) -> int:
    category, expected = classify(source.stem)
    written = 0

    with wave.open(str(source), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        total_frames = audio.getnframes()

        if (channels, sample_width, sample_rate) != (1, 2, 16000):
            raise ValueError(
                f"{source} is not 16 kHz mono 16-bit PCM: "
                f"channels={channels}, width={sample_width}, rate={sample_rate}"
            )

        for seconds in WINDOW_SECONDS:
            frames_per_window = round(sample_rate * seconds)
            full_windows = total_frames // frames_per_window
            destination = output_root / window_tag(seconds) / source.stem
            destination.mkdir(parents=True, exist_ok=True)

            audio.rewind()
            for index in range(full_windows):
                frames = audio.readframes(frames_per_window)
                output = destination / f"window_{index:04d}.wav"
                with wave.open(str(output), "wb") as clip:
                    clip.setnchannels(channels)
                    clip.setsampwidth(sample_width)
                    clip.setframerate(sample_rate)
                    clip.writeframes(frames)

                start = index * seconds
                writer.writerow(
                    {
                        "window_path": output.as_posix(),
                        "source_file": source.name,
                        "category": category,
                        "expected": expected,
                        "window_seconds": f"{seconds:g}",
                        "start_seconds": f"{start:.3f}",
                        "end_seconds": f"{start + seconds:.3f}",
                    }
                )
                written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("recordings/normalized"))
    parser.add_argument("--output", type=Path, default=Path("recordings/windows"))
    parser.add_argument("--manifest", type=Path, default=Path("recordings/windows.csv"))
    args = parser.parse_args()

    sources = sorted(
        path for path in args.input.glob("*.wav") if not path.stem.startswith("enroll_")
    )
    if not sources:
        raise SystemExit(f"No test WAV files found in {args.input}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "window_path",
        "source_file",
        "category",
        "expected",
        "window_seconds",
        "start_seconds",
        "end_seconds",
    )
    with args.manifest.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fields)
        writer.writeheader()
        total = sum(split_file(source, args.output, writer) for source in sources)

    print(f"Created {total} windows from {len(sources)} test recordings")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
