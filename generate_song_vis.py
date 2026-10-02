#!/usr/bin/env python3

"""
Convert four Demucs stem-analysis JSON files into:

1. A readable JSON file containing only data required by the LED visualizer.
2. A compact binary file intended for storage/playback on an ESP32.

The "other" stem must come from other_agreement_compiler.py
(others_agreement_v4.json, format "others_agreement_v4").

Binary frame format: 10 bytes per frame
16-bit values are little-endian (low byte first), matching the ESP32.

    Byte 0      Bass volume          0-255

    Byte 1      Drums:
                    bit 0     Kick present
                    bit 1     Hi-hat present
                    bits 2-4  Kick intensity   0-7
                    bits 5-7  Hi-hat intensity 0-7

    Bytes 2-3   Other note positions (uint16):
                    bit n     1 = note lit at frequency band n this frame
                              (bit 0 = 40-56Hz ... bit 15 = 7082-10000Hz)

    Bytes 4-5   Other note colour (uint16, RGB555):
                    bits 0-4    Red   0-31
                    bits 5-9    Green 0-31
                    bits 10-14  Blue  0-31
                    bit 15      Reserved (0)

    Bytes 6-7   Vocal colour (uint16, RGB555, same layout as other colour)

    Byte 8      Vocal volume         0-255

    Byte 9      Vocal onset          0-255

Colour: the lowest 15 frequency bands are split into three groups of 5.
Each group's band values (0-1) are summed (0-5) and scaled to 0-31:
bands 0-4 -> red, bands 5-9 -> green, bands 10-14 -> blue. The highest
band (7082-10000Hz) is not used.

Usage:

    python generate_song_vis.py song_name

Optional:

    python generate_song_vis.py song_name \
        --json-output song_led_data.json \
        --binary-output song_led_data.bin
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

SONG_PATH = Path("separated/htdemucs_ft")

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


# -----------------------------------------------------------------------------
# Colour configuration (other and vocals)
# -----------------------------------------------------------------------------

# Bands summed for each colour channel. The highest band is unused.
RED_BANDS = INPUT_FREQUENCY_BANDS[0:5]
GREEN_BANDS = INPUT_FREQUENCY_BANDS[5:10]
BLUE_BANDS = INPUT_FREQUENCY_BANDS[10:15]

# Largest 5-bit channel value. A channel whose 5 bands are all 1.0
# (sum 5.0) becomes this value, so the scale factor is 31 / 5 = 6.2.
COLOR_CHANNEL_MAX = 31


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
# Other note output
# -----------------------------------------------------------------------------

# Input file for the "other" stem, produced by other_agreement_compiler.py.
OTHER_INPUT_FILENAME = "others_agreement_v4.json"
OTHER_EXPECTED_FORMAT = "others_agreement_v4"

# For a detected note, the band with the largest increase over the next
# OTHER_LOOKAHEAD_FRAMES frames (compared with the note frame) is lit.
OTHER_LOOKAHEAD_FRAMES = 3

# Which spectrum the other note colour is calculated from:
#
# True:  each band's peak over the note frame and the lookahead frames.
#        The note frame is usually just BEFORE the note's rise, so this
#        colours the note by the sound that is arriving.
#
# False: the note frame's own spectrum only.
OTHER_COLOR_USE_LOOKAHEAD_PEAK = True


# -----------------------------------------------------------------------------
# Vocal output
# -----------------------------------------------------------------------------

# Applied to vocal volume and onset bytes.
VOCAL_GAIN = 1.0
VOCAL_GAMMA = 0.75


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

DEFAULT_JSON_OUTPUT = Path("led_visualisation_data.json")
DEFAULT_BINARY_OUTPUT = Path("led_visualisation_data.bin")

BYTES_PER_FRAME = 10

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


# -----------------------------------------------------------------------------
# Colour
# -----------------------------------------------------------------------------


def spectrum_to_rgb(spectrum: dict[str, float]) -> tuple[int, int, int]:
    """
    Convert a 16-band spectrum (values 0-1) to 5-bit R, G, B values (0-31).

    Each channel sums its 5 bands (0-5) and scales the sum to 0-31.
    """

    def channel(bands: list[str]) -> int:
        total = sum(
            clamp(spectrum.get(band, 0.0), 0.0, 1.0)
            for band in bands
        )

        return int(clamp(
            round(total / len(bands) * COLOR_CHANNEL_MAX),
            0,
            COLOR_CHANNEL_MAX,
        ))

    return channel(RED_BANDS), channel(GREEN_BANDS), channel(BLUE_BANDS)


def encode_rgb555(red: int, green: int, blue: int) -> int:
    """
    Pack 5-bit colour channels into a 16-bit value:

        bits 0-4    red
        bits 5-9    green
        bits 10-14  blue
        bit 15      reserved (0)
    """

    return (
        (int(red) & 0b11111)
        | (int(green) & 0b11111) << 5
        | (int(blue) & 0b11111) << 10
    )


def frame_spectrum(frame: dict) -> dict[str, float]:
    return {
        band: get_frequency(frame, band)
        for band in INPUT_FREQUENCY_BANDS
    }


# -----------------------------------------------------------------------------
# Other
# -----------------------------------------------------------------------------


def calculate_other_notes(frames: list[dict]) -> list[dict]:
    """
    For each frame with note_detected true:

      * position: the band with the largest increase from this frame to its
        peak over the next OTHER_LOOKAHEAD_FRAMES frames. If no band rises,
        the band with the highest peak level is used instead.
      * colour: spectrum_to_rgb() of the note's spectrum (see
        OTHER_COLOR_USE_LOOKAHEAD_PEAK).

    Frames without a note have no position bits set and colour 0.
    """

    output: list[dict] = []

    for i, frame in enumerate(frames):
        if not frame.get("note_detected", False):
            output.append({
                "note_detected": False,
                "position": None,
                "position_bits": 0,
                "rgb": (0, 0, 0),
                "color_bits": 0,
            })
            continue

        current = frame_spectrum(frame)
        lookahead = [
            frame_spectrum(candidate)
            for candidate in frames[i + 1:i + 1 + OTHER_LOOKAHEAD_FRAMES]
        ]

        # At the end of the song there may be no future frames.
        if not lookahead:
            lookahead = [current]

        future_peak = {
            band: max(spectrum[band] for spectrum in lookahead)
            for band in INPUT_FREQUENCY_BANDS
        }

        increases = [
            future_peak[band] - current[band]
            for band in INPUT_FREQUENCY_BANDS
        ]

        if max(increases) > 0.0:
            scores = increases
        else:
            scores = [future_peak[band] for band in INPUT_FREQUENCY_BANDS]

        # max() returns the first index on ties, so the result is
        # deterministic.
        position = max(range(len(scores)), key=lambda index: scores[index])

        if OTHER_COLOR_USE_LOOKAHEAD_PEAK:
            color_spectrum = {
                band: max(current[band], future_peak[band])
                for band in INPUT_FREQUENCY_BANDS
            }
        else:
            color_spectrum = current

        rgb = spectrum_to_rgb(color_spectrum)

        output.append({
            "note_detected": True,
            "position": position,
            "position_bits": 1 << position,
            "rgb": rgb,
            "color_bits": encode_rgb555(*rgb),
        })

    return output


# -----------------------------------------------------------------------------
# Drums
# -----------------------------------------------------------------------------


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


def rgb_json(rgb: tuple[int, int, int], bits: int) -> dict:
    return {
        "r": rgb[0],
        "g": rgb[1],
        "b": rgb[2],
        "bits": bits,
    }


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
    other_notes = calculate_other_notes(other_frames)

    frame_interval = estimate_frame_interval(bass_frames)

    readable_frames = []
    binary_data = bytearray()

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

        other_note = other_notes[i]

        # ---------------------------------------------------------------------
        # Vocals
        # ---------------------------------------------------------------------

        vocal_rgb = spectrum_to_rgb(frame_spectrum(vocal_frames[i]))
        vocal_color_bits = encode_rgb555(*vocal_rgb)

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
                "note_detected": other_note["note_detected"],
                "position": other_note["position"],
                "position_bits": other_note["position_bits"],
                "color": rgb_json(
                    other_note["rgb"],
                    other_note["color_bits"],
                ),
            },

            "vocals": {
                "color": rgb_json(vocal_rgb, vocal_color_bits),
                "volume": vocal_volume,
                "onset": vocal_onset,
            },
        }

        readable_frames.append(readable_frame)

        # ---------------------------------------------------------------------
        # Binary representation
        # ---------------------------------------------------------------------

        binary_data.append(bass_byte)
        binary_data.append(drum_byte)
        binary_data.extend(other_note["position_bits"].to_bytes(2, "little"))
        binary_data.extend(other_note["color_bits"].to_bytes(2, "little"))
        binary_data.extend(vocal_color_bits.to_bytes(2, "little"))
        binary_data.append(vocal_volume)
        binary_data.append(vocal_onset)

    rgb555_layout = {
        "bits_0_4": "red_0_to_31",
        "bits_5_9": "green_0_to_31",
        "bits_10_14": "blue_0_to_31",
        "bit_15": "reserved",
    }

    output = {
        "format": {
            "bytes_per_frame": BYTES_PER_FRAME,
            "byte_order": "little_endian",

            "binary_layout": {
                "byte_0": "bass_volume",
                "byte_1": "drums",
                "bytes_2_3": "other_note_positions_uint16",
                "bytes_4_5": "other_note_color_rgb555",
                "bytes_6_7": "vocal_color_rgb555",
                "byte_8": "vocal_volume",
                "byte_9": "vocal_onset",
            },

            "drum_byte": {
                "bit_0": "kick_present",
                "bit_1": "hihat_present",
                "bits_2_4": "kick_intensity_0_to_7",
                "bits_5_7": "hihat_intensity_0_to_7",
            },

            "other_note_positions": {
                f"bit_{index}": band
                for index, band in enumerate(INPUT_FREQUENCY_BANDS)
            },

            "other_note_color": rgb555_layout,
            "vocal_color": rgb555_layout,

            "color_bands": {
                "red": RED_BANDS,
                "green": GREEN_BANDS,
                "blue": BLUE_BANDS,
            },

            "other_source": {
                "format": other_data.get("format"),
                "note_detection": other_data.get("note_detection", {}),
                "position_lookahead_frames": OTHER_LOOKAHEAD_FRAMES,
                "color_uses_lookahead_peak": OTHER_COLOR_USE_LOOKAHEAD_PEAK,
            },
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

        "other_note_count": sum(
            1 for note in other_notes if note["note_detected"]
        ),

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

    other_path = song_directory / OTHER_INPUT_FILENAME

    if not other_path.exists():
        raise SystemExit(
            f"Error: {other_path} not found. "
            f"Run other_agreement_compiler.py on this song's other.mp3 first."
        )

    other_data, other_frames = load_analysis(other_path)
    other_format = other_data.get("format")

    if other_format != OTHER_EXPECTED_FORMAT:
        raise SystemExit(
            f"Error: {other_path} has format {other_format!r}, "
            f"expected {OTHER_EXPECTED_FORMAT!r}."
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

    print(
        f"Other notes:     "
        f"{output['other_note_count']}"
    )

    print()
    print(f"JSON:   {json_output_path}")
    print(f"Binary: {binary_output_path}")


if __name__ == "__main__":
    main()
