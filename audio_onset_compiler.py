#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import librosa
import numpy as np


FRAME_RATE = 25.0
FRAME_DURATION = 1.0 / FRAME_RATE

MIN_FREQUENCY = 40.0
MAX_FREQUENCY = 10000.0

FREQUENCY_BANDS = [
    (40, 56),
    (56, 80),
    (80, 113),
    (113, 159),
    (159, 225),
    (225, 317),
    (317, 448),
    (448, 632),
    (632, 893),
    (893, 1261),
    (1261, 1781),
    (1781, 2515),
    (2515, 3551),
    (3551, 5015),
    (5015, 7082),
    (7082, 10000),
]

FFT_WINDOW_SECONDS = 0.080
NORMALIZATION_PERCENTILE = 95.0

NOTE_RISE_AMOUNT = 0.30
NOTE_LOOKAHEAD_FRAMES = 2

# Time constant (seconds) of the low-pass filter applied to the onset
# envelope before note detection. Larger values suppress more noise and
# brief flickers. 0 disables the filter.
NOTE_FILTER_TIME_CONSTANT = 0.00


def robust_normalize(values, percentile=95.0):
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    positive = values[values > 0.0]

    if positive.size == 0:
        return np.zeros_like(values)

    scale = np.percentile(positive, percentile)

    if not np.isfinite(scale) or scale <= 0.0:
        scale = np.max(positive)

    if scale <= 0.0:
        return np.zeros_like(values)

    return np.clip(values / scale, 0.0, 1.0)


def make_band_labels():
    return [f"{low}-{high}Hz" for low, high in FREQUENCY_BANDS]


def calculate_onset_data(y, sample_rate):
    hop_length = max(1, int(round(sample_rate * FRAME_DURATION)))

    target_fft_samples = sample_rate * FFT_WINDOW_SECONDS
    n_fft = 2 ** int(np.ceil(np.log2(target_fft_samples)))

    magnitude = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            center=True,
        )
    )

    power = magnitude ** 2

    log_power = librosa.power_to_db(
        power,
        ref=np.max,
    )

    frequencies = librosa.fft_frequencies(
        sr=sample_rate,
        n_fft=n_fft,
    )

    overall_mask = (
        (frequencies >= MIN_FREQUENCY)
        & (frequencies < MAX_FREQUENCY)
    )

    if not np.any(overall_mask):
        raise ValueError("No FFT bins found inside 40-10000 Hz.")

    overall_spectrogram = log_power[overall_mask, :]

    overall_raw = librosa.onset.onset_strength(
        sr=sample_rate,
        S=overall_spectrogram,
        hop_length=hop_length,
        lag=1,
        max_size=1,
        center=False,
        aggregate=np.mean,
    )

    band_raw = []

    for low_hz, high_hz in FREQUENCY_BANDS:
        mask = (
            (frequencies >= low_hz)
            & (frequencies < high_hz)
        )

        if not np.any(mask):
            band_onset = np.zeros(
                log_power.shape[1],
                dtype=np.float64,
            )
        else:
            band_spectrogram = log_power[mask, :]

            band_onset = librosa.onset.onset_strength(
                sr=sample_rate,
                S=band_spectrogram,
                hop_length=hop_length,
                lag=1,
                max_size=1,
                center=False,
                aggregate=np.mean,
            )

        band_raw.append(band_onset)

    band_raw = np.asarray(
        band_raw,
        dtype=np.float64,
    )

    frame_count = min(
        len(overall_raw),
        band_raw.shape[1],
    )

    overall_raw = overall_raw[:frame_count]
    band_raw = band_raw[:, :frame_count]

    overall_normalized = robust_normalize(
        overall_raw,
        NORMALIZATION_PERCENTILE,
    )

    # One shared normalization across all 16 bands.
    band_normalized = robust_normalize(
        band_raw,
        NORMALIZATION_PERCENTILE,
    )

    timestamps = (
        np.arange(frame_count, dtype=np.float64)
        * hop_length
        / sample_rate
    )

    return (
        timestamps,
        overall_normalized,
        band_normalized,
        hop_length,
        n_fft,
    )


def low_pass_filter(values, time_constant, frame_duration):
    # First-order (RC) low-pass filter. Causal, so the start of a rise
    # stays on the same frame; only its slope and brief spikes are damped.
    values = np.asarray(values, dtype=np.float64)

    if time_constant <= 0.0 or values.size == 0:
        return values.copy()

    alpha = 1.0 - np.exp(-frame_duration / time_constant)

    filtered = np.empty_like(values)
    filtered[0] = values[0]

    for i in range(1, len(values)):
        filtered[i] = filtered[i - 1] + alpha * (values[i] - filtered[i - 1])

    return filtered


def detect_note_starts(
    overall_onset,
    rise_amount,
    lookahead_frames,
):
    detected = np.zeros(
        len(overall_onset),
        dtype=bool,
    )

    for i in range(len(overall_onset)):
        current = overall_onset[i]

        search_end = min(
            len(overall_onset),
            i + lookahead_frames + 1,
        )

        required = current + rise_amount

        for future_i in range(i + 1, search_end):
            if overall_onset[future_i] >= required:
                detected[i] = True
                break

    return detected


def compile_audio(
    input_path,
    output_path,
    rise_amount,
    lookahead_frames,
    filter_time_constant,
):
    print(f"Loading: {input_path}")

    y, sample_rate = librosa.load(
        input_path,
        sr=None,
        mono=True,
    )

    duration = len(y) / sample_rate

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Analysis rate: {FRAME_RATE:.1f} Hz")

    (
        timestamps,
        overall_onset,
        band_onsets,
        hop_length,
        n_fft,
    ) = calculate_onset_data(
        y,
        sample_rate,
    )

    filtered_onset = low_pass_filter(
        overall_onset,
        filter_time_constant,
        FRAME_DURATION,
    )

    note_detected = detect_note_starts(
        filtered_onset,
        rise_amount,
        lookahead_frames,
    )

    band_labels = make_band_labels()

    frames = []

    for i, timestamp in enumerate(timestamps):
        onset_bands = {
            label: round(
                float(band_onsets[band_index, i]),
                4,
            )
            for band_index, label in enumerate(band_labels)
        }

        frames.append(
            {
                "time": round(float(timestamp), 5),
                "onset": round(float(overall_onset[i]), 4),
                "note_detected": bool(note_detected[i]),
                "onset_bands": onset_bands,
            }
        )

    output = {
        "format": "led_audio_onsets_v1",
        "source": input_path.name,
        "duration": round(float(duration), 5),
        "sample_rate": int(sample_rate),
        "frame_rate": FRAME_RATE,
        "frame_duration": FRAME_DURATION,
        "hop_length": int(hop_length),
        "n_fft": int(n_fft),
        "frequency_range_hz": [
            MIN_FREQUENCY,
            MAX_FREQUENCY,
        ],
        "frequency_bands_hz": [
            [low, high]
            for low, high in FREQUENCY_BANDS
        ],
        "normalization": {
            "overall_onset_percentile": NORMALIZATION_PERCENTILE,
            "band_onset_percentile": NORMALIZATION_PERCENTILE,
            "band_normalization": "single shared scale across all 16 bands",
        },
        "note_detection": {
            "method": "lookahead_onset_rise",
            "rise_amount": rise_amount,
            "lookahead_frames": lookahead_frames,
            "lookahead_seconds": round(
                lookahead_frames * FRAME_DURATION,
                4,
            ),
            "filter": "first_order_low_pass",
            "filter_time_constant": filter_time_constant,
        },
        "frames": frames,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(f"Wrote: {output_path}")
    print(f"Frames: {len(frames)}")
    print(f"FFT size: {n_fft}")
    print(f"Frequency bands: {len(FREQUENCY_BANDS)}")
    print(f"Detected note-start frames: {int(np.sum(note_detected))}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert MP3 audio to 25 Hz overall onset, "
            "note-detection, and 16-band onset JSON."
        )
    )

    parser.add_argument(
        "input",
        help="Input MP3 file.",
    )

    parser.add_argument(
        "output",
        nargs="?",
        help="Output JSON. Default: <input>_onsets.json",
    )

    parser.add_argument(
        "--rise",
        type=float,
        default=NOTE_RISE_AMOUNT,
        help=f"Required onset rise. Default: {NOTE_RISE_AMOUNT}",
    )

    parser.add_argument(
        "--lookahead",
        type=int,
        default=NOTE_LOOKAHEAD_FRAMES,
        help=f"Future frames to inspect. Default: {NOTE_LOOKAHEAD_FRAMES}",
    )

    parser.add_argument(
        "--filter-tau",
        type=float,
        default=NOTE_FILTER_TIME_CONSTANT,
        help=(
            "Low-pass filter time constant in seconds for note detection. "
            f"0 disables. Default: {NOTE_FILTER_TIME_CONSTANT}"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(
            f"Input file not found: {input_path}"
        )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_name(
            input_path.stem + "_onsets.json"
        )

    if args.rise < 0.0:
        raise SystemExit("--rise must be zero or greater.")

    if args.lookahead < 1:
        raise SystemExit("--lookahead must be at least 1.")

    if args.filter_tau < 0.0:
        raise SystemExit("--filter-tau must be zero or greater.")

    compile_audio(
        input_path=input_path,
        output_path=output_path,
        rise_amount=args.rise,
        lookahead_frames=args.lookahead,
        filter_time_constant=args.filter_tau,
    )


if __name__ == "__main__":
    main()
