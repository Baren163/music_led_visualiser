#!/usr/bin/env python3

"""
Convert four Demucs stem-analysis JSON files into:

1. A readable JSON file containing only data required by the LED visualizer.
2. A compact binary file intended for storage/playback on an ESP32.

Binary frame format: 12 bytes per frame

    Byte 0      Bass volume          0-255

    Byte 1      Drums:
                    bit 0     Kick present
                    bit 1     Hi-hat present
                    bits 2-4  Kick intensity   0-7
                    bits 5-7  Hi-hat intensity 0-7

    Byte 2      Other note brightness 0-255

    Byte 3      Other note metadata:
                    bit 0     Note present
                    bits 1-4  Position 0-13
                    bits 5-7  Reserved

    Bytes 4-11  Vocal spectrum       8 x uint8

Usage:

    python prepare_led_data.py song_name

Optional:

    python prepare_led_data.py song_name \
        --json-output song_led_data.json \
        --binary-output song_led_data.bin
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

SONG_PATH = Path("separated/htdemucs")

# =============================================================================
# CONFIGURATION
# =============================================================================


# -----------------------------------------------------------------------------
# Spectrum configuration
# -----------------------------------------------------------------------------

# Frequency bands expected in each input JSON.
INPUT_FREQUENCY_BANDS = [
    "40-56Hz",
    "56-80Hz",
    "80-113Hz",
    "113-159Hz",
    "159-225Hz",
    "225-317Hz",
    "317-448Hz",
    "448-632Hz",
    "632-893Hz",
    "893-1261Hz",
    "1261-1781Hz",
    "1781-2515Hz",
    "2515-3551Hz",
    "3551-5015Hz",
    "5015-7082Hz",
    "7082-10000Hz",
]


# Convert 16 input bands into 8 output bands by combining adjacent pairs.
OUTPUT_FREQUENCY_BANDS = [
    ("40-80Hz",       ("40-56Hz", "56-80Hz")),
    ("80-159Hz",      ("80-113Hz", "113-159Hz")),
    ("159-317Hz",     ("159-225Hz", "225-317Hz")),
    ("317-632Hz",     ("317-448Hz", "448-632Hz")),
    ("632-1261Hz",    ("632-893Hz", "893-1261Hz")),
    ("1261-2515Hz",   ("1261-1781Hz", "1781-2515Hz")),
    ("2515-5015Hz",   ("2515-3551Hz", "3551-5015Hz")),
    ("5015-10000Hz",  ("5015-7082Hz", "7082-10000Hz")),
]


# How adjacent bands are combined.
#
# "max":
#     Preserves strong spectral peaks and generally works well visually.
#
# "average":
#     Produces a smoother spectrum.
#
SPECTRUM_PAIR_MODE = "max"


# -----------------------------------------------------------------------------
# Bass configuration
# -----------------------------------------------------------------------------

# Bass uses the "volume" value directly.
#
# Values are expected to already be approximately in the 0.0-1.0 range.
BASS_GAIN = 1.0

# Optional gamma adjustment.
#
# < 1.0 makes quiet bass more visible.
# > 1.0 suppresses quiet bass.
BASS_GAMMA = 0.75


# -----------------------------------------------------------------------------
# Drum frequency regions
# -----------------------------------------------------------------------------

# Kick energy comes from everything below 893 Hz.
KICK_BANDS = [
    "40-56Hz",
    "56-80Hz",
    "80-113Hz",
    "113-159Hz",
    "159-225Hz",
    "225-317Hz",
    "317-448Hz",
    "448-632Hz",
    "632-893Hz",
]

# Hi-hat energy comes from everything above 3551 Hz.
HIHAT_BANDS = [
    "3551-5015Hz",
    "5015-7082Hz",
    "7082-10000Hz",
]


# -----------------------------------------------------------------------------
# Drum transient detector
# -----------------------------------------------------------------------------

# Controls how quickly the background energy estimate follows the signal.
#
# Smaller = slower baseline = more sensitive to transients.
# Larger  = faster baseline = less sensitive.
DRUM_BASELINE_ALPHA = 0.08


# Adaptive threshold:
#
# threshold = median(novelty) + MAD * sensitivity
#
# Higher numbers produce fewer detections.
KICK_THRESHOLD_SENSITIVITY = 3.0
HIHAT_THRESHOLD_SENSITIVITY = 3.0


# Prevent extremely tiny fluctuations from being classified as beats.
KICK_MINIMUM_NOVELTY = 0.003
HIHAT_MINIMUM_NOVELTY = 0.002


# Minimum amount of time between two detections of the same drum type.
KICK_COOLDOWN_SECONDS = 0.080
HIHAT_COOLDOWN_SECONDS = 0.040


# Determines how drum intensity is normalized.
#
# The 95th percentile transient is treated as approximately maximum intensity.
DRUM_INTENSITY_PERCENTILE = 95.0


# -----------------------------------------------------------------------------
# Other onset/note output
# -----------------------------------------------------------------------------

# Input file containing the onset-based analysis for the "other" stem.
OTHER_INPUT_FILENAME = "other_onsets.json"

# Expected format marker in that file.
OTHER_EXPECTED_FORMAT = "led_audio_onsets_v1"

# Number of adjacent input frequency bands used to choose the note position.
# With 16 input bands and a 3-band window, this gives 14 possible positions.
OTHER_POSITION_WINDOW_BANDS = 3

# Brightness mapping for detected notes. The input onset is expected to be 0-1.
# Gain is applied first, then gamma. A detected note is always at least the
# configured minimum brightness so weak detections remain visible.
OTHER_BRIGHTNESS_GAIN = 1.0
OTHER_BRIGHTNESS_GAMMA = 0.75
OTHER_MIN_BRIGHTNESS_BYTE = 48
OTHER_MAX_BRIGHTNESS_BYTE = 255

# If True, the detection frame itself is included together with the following
# `lookahead_frames` frames. For example, lookahead_frames=2 examines i, i+1, i+2.
OTHER_INCLUDE_DETECTION_FRAME = True

# The input JSON's note_detection.lookahead_frames normally controls the window.
# Set this to an integer to override the JSON value for tuning, or leave as None.
OTHER_LOOKAHEAD_FRAMES_OVERRIDE: int | None = None

# Vocals retain the existing 16-to-8-band spectrum representation.
VOCAL_GAIN = 1.0
VOCAL_GAMMA = 0.75


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

DEFAULT_JSON_OUTPUT = Path("led_visualisation_data.json")
DEFAULT_BINARY_OUTPUT = Path("led_visualisation_data.bin")

JSON_INDENT = 2


# =============================================================================
# END CONFIGURATION
# =============================================================================


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def to_uint8(value: float, gain: float = 1.0, gamma: float = 1.0) -> int:
    """
    Convert an approximately 0-1 value to an unsigned byte.
    """

    value = clamp(float(value) * gain, 0.0, 1.0)

    if gamma != 1.0:
        value = value ** gamma

    return round(value * 255.0)


def load_frames(path: Path) -> list[dict]:
    """
    Load either:

        [
            {...},
            {...}
        ]

    or:

        {
            "frames": [
                {...},
                {...}
            ]
        }
    """

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        frames = data

    elif isinstance(data, dict) and isinstance(data.get("frames"), list):
        frames = data["frames"]

    else:
        raise ValueError(
            f"{path} must contain either a JSON array or an object "
            f"with a 'frames' array."
        )

    return frames


def load_analysis(path: Path) -> tuple[dict, list[dict]]:
    """Load a header-based analysis JSON while preserving its metadata."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{path} must contain a 'frames' array.")

    return data, frames


def get_frequency(frame: dict, band: str) -> float:
    frequencies = frame.get("frequencies", {})

    try:
        return float(frequencies.get(band, 0.0))
    except (TypeError, ValueError):
        return 0.0


def frequency_region_energy(
    frame: dict,
    bands: list[str],
) -> float:
    """
    Calculate average energy over several frequency bands.

    Average is used instead of sum so that regions containing different
    numbers of bands remain approximately comparable.
    """

    if not bands:
        return 0.0

    values = [
        max(0.0, get_frequency(frame, band))
        for band in bands
    ]

    return sum(values) / len(values)


def reduce_spectrum_to_8(frame: dict) -> list[float]:
    """
    Convert the original 16 frequency bands into 8 bands.
    """

    result = []

    for _, source_bands in OUTPUT_FREQUENCY_BANDS:
        values = [
            get_frequency(frame, band)
            for band in source_bands
        ]

        if SPECTRUM_PAIR_MODE == "max":
            value = max(values)

        elif SPECTRUM_PAIR_MODE == "average":
            value = sum(values) / len(values)

        else:
            raise ValueError(
                f"Unknown SPECTRUM_PAIR_MODE: "
                f"{SPECTRUM_PAIR_MODE}"
            )

        result.append(value)

    return result


def get_other_onset_band(frame: dict, band: str) -> float:
    onset_bands = frame.get("onset_bands", {})

    try:
        return max(0.0, float(onset_bands.get(band, 0.0)))
    except (TypeError, ValueError):
        return 0.0


def calculate_other_note_frames(
    frames: list[dict],
    lookahead_frames: int,
) -> list[dict]:
    """
    Convert onset-based "other" analysis into one compact note event per frame.

    For each frame where note_detected is true:
      * inspect the detection frame plus the configured lookahead window;
      * use the maximum overall onset as note brightness;
      * sum each of the 16 onset bands over that window;
      * find the strongest adjacent rolling group;
      * store the rolling-window index (0-13 for a 3-band window) as position.
    """

    if OTHER_POSITION_WINDOW_BANDS <= 0:
        raise ValueError("OTHER_POSITION_WINDOW_BANDS must be positive.")

    if OTHER_POSITION_WINDOW_BANDS > len(INPUT_FREQUENCY_BANDS):
        raise ValueError(
            "OTHER_POSITION_WINDOW_BANDS cannot exceed the number of input bands."
        )

    position_count = (
        len(INPUT_FREQUENCY_BANDS) - OTHER_POSITION_WINDOW_BANDS + 1
    )

    if position_count > 16:
        raise ValueError("Other note position must fit in 4 bits.")

    lookahead_frames = max(0, int(lookahead_frames))
    output: list[dict] = []

    for i, frame in enumerate(frames):
        note_detected = bool(frame.get("note_detected", False))

        if not note_detected:
            output.append({
                "note_detected": False,
                "brightness": 0,
                "position": 0,
            })
            continue

        if OTHER_INCLUDE_DETECTION_FRAME:
            window_start = i
            window_end = min(len(frames), i + lookahead_frames + 1)
        else:
            window_start = min(len(frames), i + 1)
            window_end = min(len(frames), i + lookahead_frames + 1)

        window = frames[window_start:window_end]

        # Ensure a detected note still has its own frame available at EOF or if
        # lookahead is configured as zero while the detection frame is excluded.
        if not window:
            window = [frame]

        peak_onset = max(
            clamp(float(candidate.get("onset", 0.0)), 0.0, 1.0)
            for candidate in window
        )

        scaled_onset = clamp(
            peak_onset * OTHER_BRIGHTNESS_GAIN,
            0.0,
            1.0,
        )

        if OTHER_BRIGHTNESS_GAMMA != 1.0:
            scaled_onset = scaled_onset ** OTHER_BRIGHTNESS_GAMMA

        brightness = round(
            OTHER_MIN_BRIGHTNESS_BYTE
            + scaled_onset
            * (OTHER_MAX_BRIGHTNESS_BYTE - OTHER_MIN_BRIGHTNESS_BYTE)
        )

        brightness = int(clamp(
            brightness,
            OTHER_MIN_BRIGHTNESS_BYTE,
            OTHER_MAX_BRIGHTNESS_BYTE,
        ))

        band_totals = []
        for band in INPUT_FREQUENCY_BANDS:
            band_totals.append(sum(
                get_other_onset_band(candidate, band)
                for candidate in window
            ))

        rolling_sums = []
        for position in range(position_count):
            rolling_sums.append(sum(
                band_totals[
                    position:position + OTHER_POSITION_WINDOW_BANDS
                ]
            ))

        # max() returns the first matching index, giving deterministic behaviour
        # if two positions have exactly equal onset sums.
        position = max(
            range(position_count),
            key=lambda index: rolling_sums[index],
        )

        output.append({
            "note_detected": True,
            "brightness": brightness,
            "position": position,
        })

    return output


def encode_other_metadata(note_detected: bool, position: int) -> int:
    """
    Other metadata byte:

        bit 0       note present
        bits 1-4    position 0-13
        bits 5-7    reserved
    """

    value = 0

    if note_detected:
        value |= 1 << 0

    value |= (int(position) & 0b1111) << 1
    return value


def exponential_novelty(
    values: list[float],
    alpha: float,
) -> list[float]:
    """
    Measure sudden increases relative to a slowly moving baseline.

    This is much more useful for detecting drum hits than simply checking
    absolute low/high-frequency energy.
    """

    if not values:
        return []

    baseline = values[0]

    novelty = [0.0]

    for value in values[1:]:

        # Compare the current frame against the OLD baseline first.
        difference = max(0.0, value - baseline)

        novelty.append(difference)

        # Then update the background level.
        baseline = (
            alpha * value
            + (1.0 - alpha) * baseline
        )

    return novelty


def median_absolute_deviation(values: list[float]) -> float:
    if not values:
        return 0.0

    median = statistics.median(values)

    deviations = [
        abs(value - median)
        for value in values
    ]

    return statistics.median(deviations)


def calculate_threshold(
    novelty: list[float],
    sensitivity: float,
    minimum_threshold: float,
) -> float:

    if not novelty:
        return minimum_threshold

    median = statistics.median(novelty)
    mad = median_absolute_deviation(novelty)

    threshold = median + sensitivity * mad

    return max(threshold, minimum_threshold)


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    position = (
        percentage / 100.0
        * (len(ordered) - 1)
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def detect_transients(
    times: list[float],
    novelty: list[float],
    threshold: float,
    cooldown_seconds: float,
) -> tuple[list[bool], list[int]]:
    """
    Find local transient peaks.

    Returns:

        detections:
            True when a beat occurs.

        intensity:
            0-7 intensity value.
    """

    count = len(novelty)

    detections = [False] * count
    intensities = [0] * count

    if count == 0:
        return detections, intensities

    intensity_reference = percentile(
        novelty,
        DRUM_INTENSITY_PERCENTILE,
    )

    intensity_reference = max(
        intensity_reference,
        threshold,
        1e-9,
    )

    last_detection_time = -float("inf")

    for i in range(count):

        current = novelty[i]

        if current < threshold:
            continue

        previous = novelty[i - 1] if i > 0 else 0.0
        following = novelty[i + 1] if i + 1 < count else 0.0

        # Only trigger on the peak of the transient.
        if current < previous or current < following:
            continue

        current_time = times[i]

        if current_time - last_detection_time < cooldown_seconds:
            continue

        detections[i] = True

        normalized = clamp(
            current / intensity_reference,
            0.0,
            1.0,
        )

        intensity = max(
            1,
            round(normalized * 7.0),
        )

        intensities[i] = intensity
        last_detection_time = current_time

    return detections, intensities


def analyse_drums(
    frames: list[dict],
) -> dict[str, list]:

    times = [
        float(frame.get("time", i))
        for i, frame in enumerate(frames)
    ]

    kick_energy = [
        frequency_region_energy(frame, KICK_BANDS)
        for frame in frames
    ]

    hihat_energy = [
        frequency_region_energy(frame, HIHAT_BANDS)
        for frame in frames
    ]

    kick_novelty = exponential_novelty(
        kick_energy,
        DRUM_BASELINE_ALPHA,
    )

    hihat_novelty = exponential_novelty(
        hihat_energy,
        DRUM_BASELINE_ALPHA,
    )

    kick_threshold = calculate_threshold(
        kick_novelty,
        KICK_THRESHOLD_SENSITIVITY,
        KICK_MINIMUM_NOVELTY,
    )

    hihat_threshold = calculate_threshold(
        hihat_novelty,
        HIHAT_THRESHOLD_SENSITIVITY,
        HIHAT_MINIMUM_NOVELTY,
    )

    kick_detected, kick_intensity = detect_transients(
        times,
        kick_novelty,
        kick_threshold,
        KICK_COOLDOWN_SECONDS,
    )

    hihat_detected, hihat_intensity = detect_transients(
        times,
        hihat_novelty,
        hihat_threshold,
        HIHAT_COOLDOWN_SECONDS,
    )

    return {
        "kick": kick_detected,
        "kick_intensity": kick_intensity,

        "hihat": hihat_detected,
        "hihat_intensity": hihat_intensity,

        "kick_energy": kick_energy,
        "hihat_energy": hihat_energy,

        "kick_novelty": kick_novelty,
        "hihat_novelty": hihat_novelty,

        "kick_threshold": kick_threshold,
        "hihat_threshold": hihat_threshold,
    }


def encode_drum_byte(
    kick: bool,
    hihat: bool,
    kick_intensity: int,
    hihat_intensity: int,
) -> int:
    """
    Drum byte:

        bit 0       kick
        bit 1       hi-hat
        bits 2-4    kick intensity
        bits 5-7    hi-hat intensity
    """

    value = 0

    if kick:
        value |= 1 << 0

    if hihat:
        value |= 1 << 1

    value |= (kick_intensity & 0b111) << 2
    value |= (hihat_intensity & 0b111) << 5

    return value


def estimate_frame_interval(frames: list[dict]) -> float | None:
    if len(frames) < 2:
        return None

    intervals = []

    for previous, current in zip(frames, frames[1:]):
        try:
            difference = (
                float(current["time"])
                - float(previous["time"])
            )

            if difference > 0:
                intervals.append(difference)

        except (KeyError, TypeError, ValueError):
            continue

    if not intervals:
        return None

    return statistics.median(intervals)


def process(
    bass_frames: list[dict],
    drum_frames: list[dict],
    other_data: dict,
    other_frames: list[dict],
    vocal_frames: list[dict],
) -> tuple[dict, bytearray]:

    lengths = {
        "bass": len(bass_frames),
        "drums": len(drum_frames),
        "other": len(other_frames),
        "vocals": len(vocal_frames),
    }

    frame_count = min(lengths.values())

    if len(set(lengths.values())) != 1:
        print(
            "Warning: input files have different frame counts:"
        )

        for name, length in lengths.items():
            print(f"  {name:7s}: {length}")

        print(
            f"Using the first {frame_count} frames from every file."
        )

    bass_frames = bass_frames[:frame_count]
    drum_frames = drum_frames[:frame_count]
    other_frames = other_frames[:frame_count]
    vocal_frames = vocal_frames[:frame_count]

    drums = analyse_drums(drum_frames)

    other_format = other_data.get("format")
    if other_format != OTHER_EXPECTED_FORMAT:
        raise ValueError(
            f"Expected other input format {OTHER_EXPECTED_FORMAT!r}, "
            f"got {other_format!r}."
        )

    note_detection = other_data.get("note_detection", {})
    json_lookahead_frames = note_detection.get("lookahead_frames", 0)

    try:
        json_lookahead_frames = int(json_lookahead_frames)
    except (TypeError, ValueError):
        raise ValueError(
            "other onset JSON note_detection.lookahead_frames must be an integer."
        )

    if OTHER_LOOKAHEAD_FRAMES_OVERRIDE is None:
        other_lookahead_frames = json_lookahead_frames
    else:
        other_lookahead_frames = OTHER_LOOKAHEAD_FRAMES_OVERRIDE

    other_note_frames = calculate_other_note_frames(
        other_frames,
        other_lookahead_frames,
    )

    frame_interval = estimate_frame_interval(bass_frames)

    readable_frames = []
    binary_data = bytearray()

    output_band_names = [
        name
        for name, _ in OUTPUT_FREQUENCY_BANDS
    ]

    for i in range(frame_count):

        time_value = float(
            bass_frames[i].get("time", i)
        )

        # ---------------------------------------------------------------------
        # Bass
        # ---------------------------------------------------------------------

        bass_volume_float = float(
            bass_frames[i].get("volume", 0.0)
        )

        bass_byte = to_uint8(
            bass_volume_float,
            BASS_GAIN,
            BASS_GAMMA,
        )

        # ---------------------------------------------------------------------
        # Drums
        # ---------------------------------------------------------------------

        kick = drums["kick"][i]
        hihat = drums["hihat"][i]

        kick_intensity = drums["kick_intensity"][i]
        hihat_intensity = drums["hihat_intensity"][i]

        drum_byte = encode_drum_byte(
            kick,
            hihat,
            kick_intensity,
            hihat_intensity,
        )

        # ---------------------------------------------------------------------
        # Other
        # ---------------------------------------------------------------------

        other_note = other_note_frames[i]
        other_note_detected = other_note["note_detected"]
        other_brightness = other_note["brightness"]
        other_position = other_note["position"]

        other_metadata_byte = encode_other_metadata(
            other_note_detected,
            other_position,
        )

        # ---------------------------------------------------------------------
        # Vocals
        # ---------------------------------------------------------------------

        vocal_values = reduce_spectrum_to_8(
            vocal_frames[i]
        )

        vocal_bytes = [
            to_uint8(
                value,
                VOCAL_GAIN,
                VOCAL_GAMMA,
            )
            for value in vocal_values
        ]

        vocal_volume = to_uint8(
            float(vocal_frames[i].get("volume", 0.0)),
            VOCAL_GAIN,
            VOCAL_GAMMA,
        )

        vocal_onset = to_uint8(
            float(vocal_frames[i].get("onset", 0.0)),
            VOCAL_GAIN,
            VOCAL_GAMMA,
        )            

        # ---------------------------------------------------------------------
        # Readable JSON representation
        # ---------------------------------------------------------------------

        readable_frame = {
            "time": round(time_value, 6),

            "bass": {
                "volume": round(bass_volume_float, 6),
                "byte": bass_byte,
            },

            "drums": {
                "kick": kick,
                "kick_intensity": kick_intensity,

                "hihat": hihat,
                "hihat_intensity": hihat_intensity,

                "byte": drum_byte,
            },

            "other": {
                "note_detected": other_note_detected,
                "brightness": other_brightness,
                "position": other_position,
                "metadata_byte": other_metadata_byte,
            },

            "vocals": {
                "spectrum": {
                    output_band_names[j]: vocal_bytes[j]
                    for j in range(8)
                },
                "volume": vocal_volume,
                "onset": vocal_onset
            },
        }

        readable_frames.append(readable_frame)

        # ---------------------------------------------------------------------
        # Binary representation
        # ---------------------------------------------------------------------

        binary_data.append(bass_byte)
        binary_data.append(drum_byte)
        binary_data.append(other_brightness)
        binary_data.append(other_metadata_byte)
        binary_data.extend(vocal_bytes)
        binary_data.append(vocal_volume)
        binary_data.append(vocal_onset)

    output = {
        "format": {
            "bytes_per_frame": 12,

            "binary_layout": {
                "byte_0": "bass_volume",
                "byte_1": "drums",
                "byte_2": "other_note_brightness",
                "byte_3": "other_note_metadata",
                "bytes_4_11": "vocal_frequency_bands",
            },

            "drum_byte": {
                "bit_0": "kick_present",
                "bit_1": "hihat_present",
                "bits_2_4": "kick_intensity_0_to_7",
                "bits_5_7": "hihat_intensity_0_to_7",
            },

            "other_metadata_byte": {
                "bit_0": "note_detected",
                "bits_1_4": "position_0_to_13",
                "bits_5_7": "reserved",
            },

            "other_position": {
                "input_band_count": len(INPUT_FREQUENCY_BANDS),
                "rolling_window_bands": OTHER_POSITION_WINDOW_BANDS,
                "position_count": (
                    len(INPUT_FREQUENCY_BANDS)
                    - OTHER_POSITION_WINDOW_BANDS
                    + 1
                ),
                "lookahead_frames": other_lookahead_frames,
                "include_detection_frame": OTHER_INCLUDE_DETECTION_FRAME,
            },

            "frequency_bands": output_band_names,
        },

        "frame_count": frame_count,

        "frame_interval_seconds": (
            round(frame_interval, 9)
            if frame_interval is not None
            else None
        ),

        "duration_seconds": (
            round(
                float(
                    bass_frames[-1].get("time", 0.0)
                ),
                6,
            )
            if bass_frames
            else 0.0
        ),

        "drum_detector": {
            "kick_threshold": round(
                drums["kick_threshold"],
                8,
            ),

            "hihat_threshold": round(
                drums["hihat_threshold"],
                8,
            ),

            "kick_count": sum(drums["kick"]),

            "hihat_count": sum(
                drums["hihat"]
            ),
        },

        "frames": readable_frames,
    }

    return output, binary_data


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Convert bass/drums/other/vocal analysis JSON files "
            "into LED visualisation data."
        )
    )

    parser.add_argument(
        "song",
        type=Path,
        help="Song name",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
    )

    parser.add_argument(
        "--binary-output",
        type=Path,
        default=DEFAULT_BINARY_OUTPUT,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_arguments()

    print("Loading input files...")

    song_directory = SONG_PATH / args.song

    bass_frames = load_frames(song_directory / "bass.json")
    drum_frames = load_frames(song_directory / "drums.json")
    other_data, other_frames = load_analysis(
        song_directory / OTHER_INPUT_FILENAME
    )
    vocal_frames = load_frames(song_directory / "vocals.json")

    print("Processing visualisation data...")

    output, binary_data = process(
        bass_frames,
        drum_frames,
        other_data,
        other_frames,
        vocal_frames,
    )

    json_output_path = song_directory / args.json_output
    binary_output_path = song_directory / args.binary_output

    json_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    binary_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with json_output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=JSON_INDENT,
        )

    with binary_output_path.open("wb") as file:
        file.write(binary_data)

    print()
    print(f"Frames:          {output['frame_count']}")
    print(f"Bytes/frame:     {output['format']['bytes_per_frame']}")
    print(f"Binary size:     {len(binary_data)} bytes")

    if output["frame_interval_seconds"] is not None:
        print(
            f"Frame interval:  "
            f"{output['frame_interval_seconds']:.6f} s"
        )

    print(
        f"Kick detections: "
        f"{output['drum_detector']['kick_count']}"
    )

    print(
        f"Hi-hat detections: "
        f"{output['drum_detector']['hihat_count']}"
    )

    print()
    print(f"JSON:   {json_output_path}")
    print(f"Binary: {binary_output_path}")


if __name__ == "__main__":
    main()