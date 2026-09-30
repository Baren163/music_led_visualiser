#!/usr/bin/env python3
"""
audio_compile_16bands_40ms.py

Convert an MP3/WAV/etc. audio file into a time-indexed stream of
audio-frequency features suitable for driving an LED display.

Changes from the earlier version:
    - 16 logarithmically spaced frequency bands instead of 8
    - Same total frequency range: 40 Hz to 10,000 Hz
    - One analysis frame every 40 ms instead of every 20 ms
    - Effective analysis/output rate: 25 frames per second

Output:
    JSON file containing:
      - metadata
      - one frame every 40 ms
      - volume (RMS, normalized 0..1)
      - 16 logarithmically spaced frequency-band levels (0..1)
      - beat/onset strength (0..1)

Dependencies:
    pip install librosa soundfile numpy

Usage:
    python audio_compile_16bands_40ms.py song.mp3
    python audio_compile_16bands_40ms.py song.mp3 output.json
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np


# 40 ms gives 25 analysis frames per second.
FRAME_DURATION = 0.040

# 16 logarithmically spaced bands over exactly the same 40 Hz to 10 kHz range.
# Log spacing is preferable here because musical pitch perception is logarithmic.
FREQUENCY_BANDS = [(40, 56), (56, 80), (80, 113), (113, 159), (159, 225), (225, 317), (317, 448), (448, 632), (632, 893), (893, 1261), (1261, 1781), (1781, 2515), (2515, 3551), (3551, 5015), (5015, 7082), (7082, 10000)]


def rms_to_01(rms):
    """Convert RMS values to a stable 0..1 logarithmic level."""
    db = librosa.amplitude_to_db(rms, ref=1.0)
    level = (db + 60.0) / 60.0
    return np.clip(level, 0.0, 1.0).astype(np.float32)


def normalize_features(values):
    """Normalize a feature array safely to 0..1."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values

    low = np.percentile(values, 5)
    high = np.percentile(values, 95)

    if high <= low:
        return np.zeros_like(values)

    return np.clip(
        (values - low) / (high - low),
        0.0,
        1.0,
    ).astype(np.float32)


def analyse_audio(input_path):
    print(f"Loading: {input_path}")

    y, sample_rate = librosa.load(
        input_path,
        sr=None,
        mono=True,
    )

    duration = len(y) / sample_rate

    # Advance the analysis window by 40 ms each frame.
    hop_length = max(
        1,
        round(sample_rate * FRAME_DURATION),
    )

    # Keep roughly an 80 ms FFT window.
    # The larger window improves frequency resolution now that we have
    # twice as many frequency bands, while frames are still emitted every 40 ms.
    n_fft = 2 ** int(
        np.ceil(
            np.log2(sample_rate * 0.080)
        )
    )

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration:    {duration:.2f} seconds")
    print(f"Frame rate:  {1 / FRAME_DURATION:.0f} FPS")
    print(f"FFT size:    {n_fft}")
    print(f"Bands:       {len(FREQUENCY_BANDS)}")

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )

    magnitude = np.abs(stft)

    frequencies = librosa.fft_frequencies(
        sr=sample_rate,
        n_fft=n_fft,
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length,
        center=True,
    )[0]

    volume = rms_to_01(rms)

    band_levels = []

    for low_hz, high_hz in FREQUENCY_BANDS:
        mask = (
            (frequencies >= low_hz)
            & (frequencies < high_hz)
        )

        if not np.any(mask):
            band_energy = np.zeros(
                magnitude.shape[1]
            )
        else:
            band_energy = np.mean(
                magnitude[mask, :],
                axis=0,
            )

        band_levels.append(
            normalize_features(band_energy)
        )

    band_levels = np.asarray(
        band_levels
    )

    onset = librosa.onset.onset_strength(
        y=y,
        sr=sample_rate,
        hop_length=hop_length,
    )

    onset = normalize_features(onset)

    frame_count = min(
        len(volume),
        band_levels.shape[1],
        len(onset),
    )

    frames = []

    for i in range(frame_count):
        time_seconds = i * FRAME_DURATION

        frames.append({
            "time": round(time_seconds, 4),
            "volume": round(
                float(volume[i]),
                4,
            ),
            "frequencies": {
                f"{low}-{high}Hz": round(
                    float(band_levels[j, i]),
                    4,
                )
                for j, (low, high)
                in enumerate(FREQUENCY_BANDS)
            },
            "onset": round(
                float(onset[i]),
                4,
            ),
        })

    return {
        "format": "led_audio_features_v2",
        "source": Path(input_path).name,
        "duration": round(duration, 4),
        "frame_rate": 1 / FRAME_DURATION,
        "frame_duration": FRAME_DURATION,
        "frequency_bands_hz": FREQUENCY_BANDS,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert an audio file into LED-friendly "
            "frequency data with 16 bands at 25 FPS."
        )
    )

    parser.add_argument(
        "input",
        help="Input MP3/WAV/etc. audio file",
    )

    parser.add_argument(
        "output",
        nargs="?",
        help=(
            "Output JSON file "
            "(default: input filename + .json)"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise SystemExit(
            f"Input file not found: {input_path}"
        )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(
            ".json"
        )

    data = analyse_audio(
        input_path
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )

    print(f"\nWrote: {output_path}")
    print(
        f"Frames: {len(data['frames'])}"
    )
    print(
        f"Approx. size: "
        f"{output_path.stat().st_size / 1024:.1f} KB"
    )


if __name__ == "__main__":
    main()
