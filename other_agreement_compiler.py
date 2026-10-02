#!/usr/bin/env python3
"""
other_agreement_compiler.py

Version 4 of the "other" note detector: a note is detected where the
volume, the overall onset and the frequency spectrum all agree.

For every frame, three evidence values are measured over the next
LOOKAHEAD_FRAMES frames:

    volume term   largest rise in volume. Volume is already on a dB scale,
                  so this is a relative rise that treats quiet and loud
                  sections alike.
    onset term    peak overall onset value. Onset strength already
                  measures an increase in energy, so its value is used
                  rather than its rise.
    band term     largest rise of any single frequency band. Band rises
                  are used rather than band levels, so sustained sounds
                  don't count as new notes.

Each term is scaled to 0-1 against its own distribution over the song, so
"medium" means the same for all three. They are combined with a weighted
geometric mean, which scores three medium values higher than one or two
high values with the rest low. A note is detected at frames where this
agreement value passes AGREEMENT_THRESHOLD and is the highest within
PEAK_RADIUS_FRAMES on either side, so one note gives one detection.

Output (next to the input file):

    others_agreement_v4.json          time, note_detected, volume,
                                      frequencies, onset
    others_agreement_debug_v4_1.json  (when WRITE_DEBUG_JSON is True)
                                      the three agreement inputs per frame

Both can be viewed with json_viewer_pygame.py.

Usage:

    python other_agreement_compiler.py separated/htdemucs/timeless/other.mp3
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------

FRAME_RATE = 25.0
FRAME_DURATION = 1.0 / FRAME_RATE

FFT_WINDOW_SECONDS = 0.080

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

# Overall onset is scaled so this percentile of its positive values is 1.0.
ONSET_NORMALIZATION_PERCENTILE = 95.0


# -----------------------------------------------------------------------------
# Note-detection agreement
# -----------------------------------------------------------------------------

# How many future frames each frame looks at (3 frames = 120 ms at 25 Hz).
LOOKAHEAD_FRAMES = 3

# Each agreement term is scaled so this percentile of its positive values
# becomes 1.0. Lower = every term reaches 1.0 more easily.
TERM_NORMALIZATION_PERCENTILE = 95.0

# Relative importance of each term in the geometric mean. Lowering a weight
# lets notes through even when that term is weak; e.g. lower VOLUME_WEIGHT
# to catch notes that change pitch without getting louder.
VOLUME_WEIGHT = 0.5
ONSET_WEIGHT = 1.0
BAND_WEIGHT = 1.0

# Each term is lifted to at least this value before combining, so a single
# zero term weakens the agreement instead of forcing it to zero.
# 0 = strict (all three must be present), higher = more forgiving.
TERM_FLOOR = 0.1

# Agreement (0-1) a frame needs to count as a note.
# Higher = fewer, surer notes.
AGREEMENT_THRESHOLD = 0.45

# A note must have the highest agreement within this many frames on either
# side, so one note can't trigger several neighbouring frames. This also
# sets the minimum gap between notes: (PEAK_RADIUS_FRAMES + 1) frames.
PEAK_RADIUS_FRAMES = 2


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

OUTPUT_FILENAME = "others_agreement_v4.json"
OUTPUT_FORMAT = "others_agreement_v4"

# Also write the three agreement inputs for tuning in json_viewer_pygame.py.
WRITE_DEBUG_JSON = True

DEBUG_OUTPUT_FILENAME = "others_agreement_debug_v4_1.json"
DEBUG_OUTPUT_FORMAT = "others_agreement_debug_v4_1"


# =============================================================================
# END CONFIGURATION
# =============================================================================


def band_labels():
    return [f"{low}-{high}Hz" for low, high in FREQUENCY_BANDS]


def rms_to_01(rms):
    """Convert RMS to a 0-1 level covering -60 dB to 0 dB."""
    db = librosa.amplitude_to_db(rms, ref=1.0)
    return np.clip((db + 60.0) / 60.0, 0.0, 1.0)


def percentile_normalize(values):
    """Normalize to 0-1 between the 5th and 95th percentiles."""
    low = np.percentile(values, 5)
    high = np.percentile(values, 95)

    if high <= low:
        return np.zeros_like(values)

    return np.clip((values - low) / (high - low), 0.0, 1.0)


def robust_scale(values, percentile):
    """
    Scale so the given percentile of the positive values becomes 1.0.
    Returns the scaled values (clipped to 0-1) and the scale used.
    """
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    positive = values[values > 0.0]

    if positive.size == 0:
        return np.zeros_like(values), 1.0

    scale = float(np.percentile(positive, percentile))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(positive))

    return np.clip(values / scale, 0.0, 1.0), scale


def calculate_features(y, sample_rate):
    """
    Volume and 16 band levels match audio_compile_16bands_40ms.py; overall
    onset matches audio_onset_compiler.py.
    """
    hop_length = max(1, int(round(sample_rate * FRAME_DURATION)))
    n_fft = 2 ** int(np.ceil(np.log2(sample_rate * FFT_WINDOW_SECONDS)))

    magnitude = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
            center=True,
        )
    )

    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

    rms = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length,
        center=True,
    )[0]

    volume = rms_to_01(rms)

    bands = []

    for low_hz, high_hz in FREQUENCY_BANDS:
        mask = (frequencies >= low_hz) & (frequencies < high_hz)

        if not np.any(mask):
            bands.append(np.zeros(magnitude.shape[1]))
        else:
            bands.append(
                percentile_normalize(np.mean(magnitude[mask, :], axis=0))
            )

    # Shape (frames, bands).
    bands = np.asarray(bands).T

    log_power = librosa.power_to_db(magnitude ** 2, ref=np.max)
    overall_mask = (frequencies >= MIN_FREQUENCY) & (frequencies < MAX_FREQUENCY)

    onset = librosa.onset.onset_strength(
        sr=sample_rate,
        S=log_power[overall_mask, :],
        hop_length=hop_length,
        lag=1,
        max_size=1,
        center=False,
        aggregate=np.mean,
    )

    onset, _ = robust_scale(onset, ONSET_NORMALIZATION_PERCENTILE)

    frame_count = min(len(volume), bands.shape[0], len(onset))

    return (
        volume[:frame_count],
        bands[:frame_count],
        onset[:frame_count],
        hop_length,
        n_fft,
    )


def future_max(values, frames):
    """Maximum over frames i+1 .. i+frames (edge-padded at the end)."""
    count = values.shape[0]
    padding = [(0, frames)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, padding, mode="edge")

    return np.max(
        np.stack([padded[k:k + count] for k in range(1, frames + 1)]),
        axis=0,
    )


def calculate_agreement(volume, bands, onset):
    volume_rise = np.maximum(future_max(volume, LOOKAHEAD_FRAMES) - volume, 0.0)
    band_rises = np.maximum(future_max(bands, LOOKAHEAD_FRAMES) - bands, 0.0)
    onset_peak = future_max(onset, LOOKAHEAD_FRAMES)

    volume_term, _ = robust_scale(volume_rise, TERM_NORMALIZATION_PERCENTILE)
    onset_term, _ = robust_scale(onset_peak, TERM_NORMALIZATION_PERCENTILE)

    # The band term is the biggest single-band rise. Every band is divided
    # by the same scale, so the tallest debug bar equals the band term.
    band_term, band_scale = robust_scale(
        np.max(band_rises, axis=1),
        TERM_NORMALIZATION_PERCENTILE,
    )
    band_changes = np.clip(band_rises / band_scale, 0.0, 1.0)

    terms = np.stack([volume_term, onset_term, band_term])
    weights = np.array([VOLUME_WEIGHT, ONSET_WEIGHT, BAND_WEIGHT])[:, None]

    lifted = TERM_FLOOR + (1.0 - TERM_FLOOR) * terms

    # Weighted geometric mean.
    agreement = np.exp(
        np.sum(weights * np.log(np.maximum(lifted, 1e-9)), axis=0)
        / max(np.sum(weights), 1e-9)
    )

    return volume_term, onset_term, band_changes, agreement


def pick_notes(agreement):
    """Frames above the threshold that are the local agreement peak."""
    count = len(agreement)
    detected = np.zeros(count, dtype=bool)

    for i in range(count):
        if agreement[i] < AGREEMENT_THRESHOLD:
            continue

        before = agreement[max(0, i - PEAK_RADIUS_FRAMES):i]
        after = agreement[i + 1:i + PEAK_RADIUS_FRAMES + 1]

        # Strictly higher than earlier frames so a flat peak triggers once.
        if np.all(before < agreement[i]) and np.all(after <= agreement[i]):
            detected[i] = True

    return detected


def detection_settings():
    return {
        "method": "volume_onset_band_agreement",
        "lookahead_frames": LOOKAHEAD_FRAMES,
        "term_normalization_percentile": TERM_NORMALIZATION_PERCENTILE,
        "weights": {
            "volume": VOLUME_WEIGHT,
            "onset": ONSET_WEIGHT,
            "band": BAND_WEIGHT,
        },
        "term_floor": TERM_FLOOR,
        "agreement_threshold": AGREEMENT_THRESHOLD,
        "peak_radius_frames": PEAK_RADIUS_FRAMES,
    }


def compile_audio(input_path):
    print(f"Loading: {input_path}")

    y, sample_rate = librosa.load(input_path, sr=None, mono=True)
    duration = len(y) / sample_rate

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")

    volume, bands, onset, hop_length, n_fft = calculate_features(y, sample_rate)
    volume_term, onset_term, band_changes, agreement = calculate_agreement(
        volume, bands, onset
    )
    detected = pick_notes(agreement)

    labels = band_labels()
    times = np.arange(len(volume)) * hop_length / sample_rate

    header = {
        "source": input_path.name,
        "duration": round(float(duration), 5),
        "sample_rate": int(sample_rate),
        "frame_rate": FRAME_RATE,
        "frame_duration": FRAME_DURATION,
        "hop_length": int(hop_length),
        "n_fft": int(n_fft),
        "frequency_bands_hz": [[low, high] for low, high in FREQUENCY_BANDS],
        "note_detection": detection_settings(),
    }

    output = {
        "format": OUTPUT_FORMAT,
        **header,
        "frames": [
            {
                "time": round(float(times[i]), 5),
                "note_detected": bool(detected[i]),
                "volume": round(float(volume[i]), 4),
                "frequencies": {
                    label: round(float(bands[i, j]), 4)
                    for j, label in enumerate(labels)
                },
                "onset": round(float(onset[i]), 4),
            }
            for i in range(len(times))
        ],
    }

    output_path = input_path.with_name(OUTPUT_FILENAME)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Wrote: {output_path}")

    if WRITE_DEBUG_JSON:
        # Every value here is the 0-1 term that feeds the agreement for
        # that frame, looking LOOKAHEAD_FRAMES ahead.
        debug_output = {
            "format": DEBUG_OUTPUT_FORMAT,
            **header,
            "frames": [
                {
                    "time": round(float(times[i]), 5),
                    "note_detected": bool(detected[i]),
                    "volume_change": round(float(volume_term[i]), 4),
                    "frequency_changes": {
                        label: round(float(band_changes[i, j]), 4)
                        for j, label in enumerate(labels)
                    },
                    "onset": round(float(onset_term[i]), 4),
                    "agreement": round(float(agreement[i]), 4),
                }
                for i in range(len(times))
            ],
        }

        debug_path = input_path.with_name(DEBUG_OUTPUT_FILENAME)

        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(debug_output, f, indent=2)

        print(f"Wrote: {debug_path}")

    note_count = int(np.sum(detected))

    print(f"Frames: {len(times)}")
    print(f"Detected notes: {note_count} ({note_count / duration:.2f} per second)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect notes in the 'other' stem where volume, onset and "
            "frequency bands agree."
        )
    )

    parser.add_argument(
        "input",
        help="Input audio file (normally the Demucs other.mp3).",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    compile_audio(input_path)


if __name__ == "__main__":
    main()
