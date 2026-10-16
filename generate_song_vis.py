#!/usr/bin/env python3

"""
Convert a song's Demucs stem analysis into LED data for the ESP32:

1. led_visualisation_data.bin   fixed-size frames for bass, drums and vocals
2. led_other_notes.bin          the "other" notes, as a list of note events
3. led_visualisation_data.json  readable version of both

"Other" notes come from one of:

    manual  other_manual.json, recorded with manual_other_recorder.py
    auto    others_agreement_v4.json, from other_agreement_compiler.py

Choose with --other manual|auto. Without it, manual is used if the song
has other_manual.json, otherwise auto; if neither exists it stops with an
error.

All multi-byte values are little-endian (low byte first), matching the
ESP32.

led_visualisation_data.bin: 6 bytes per frame, 25 frames per second.
Frame N starts at byte N * 6.

    Byte 0      Bass volume          0-255

    Byte 1      Drums:
                    bit 0     Kick present
                    bit 1     Hi-hat present
                    bits 2-4  Kick intensity   0-7
                    bits 5-7  Hi-hat intensity 0-7

    Bytes 2-3   Vocal colour (uint16, RGB555):
                    bits 0-4    Red   0-31
                    bits 5-9    Green 0-31
                    bits 10-14  Blue  0-31
                    bit 15      Reserved (0)

    Byte 4      Vocal volume         0-255

    Byte 5      Vocal onset          0-255

led_other_notes.bin: 7 bytes per note, sorted by frame, then LED.
Note K starts at byte K * 7; the note count is file size / 7.

    Bytes 0-1   Frame the note starts on (uint16)
    Byte 2      LED position 0-47 within the other section
    Byte 3      Red   0-255
    Byte 4      Green 0-255
    Byte 5      Blue  0-255
    Byte 6      Fade  0-255: each frame after the note, the LED's
                brightness is multiplied by (255 - fade) / 255.
                0 holds the LED until another note replaces it.

Vocal colour: the lowest 15 frequency bands are split into three groups of
5. Each group's band values (0-1) are summed (0-5) and scaled to 0-31:
bands 0-4 -> red, bands 5-9 -> green, bands 10-14 -> blue. The highest
band (7082-10000Hz) is not used. Automatic "other" notes are coloured the
same way and scaled to 0-255.

Usage:

    python generate_song_vis.py song_name

Optional:

    python generate_song_vis.py song_name --other auto \
        --json-output song_led_data.json \
        --binary-output song_led_data.bin \
        --other-binary-output song_other_notes.bin
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
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

# Manual notes, recorded with manual_other_recorder.py.
OTHER_MANUAL_FILENAME = "other_manual.json"
OTHER_MANUAL_FORMAT = "other_manual_v1"

# Automatic detection, produced by other_agreement_compiler.py.
OTHER_INPUT_FILENAME = "others_agreement_v4.json"
OTHER_EXPECTED_FORMAT = "others_agreement_v4"

OTHER_LED_COUNT = 48

# Automatic notes are detected per frequency band (16 bands). Each band
# lights this many neighbouring LEDs: 48 LEDs / 16 bands = 3.
AUTO_LEDS_PER_BAND = OTHER_LED_COUNT // 16

# Fade given to automatic notes (see the note format above).
AUTO_NOTE_FADE = 40

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
DEFAULT_OTHER_BINARY_OUTPUT = Path("led_other_notes.bin")

BYTES_PER_FRAME = 6

# Frame (uint16), position, red, green, blue, fade.
OTHER_NOTE_STRUCT = struct.Struct("<HBBBBB")
BYTES_PER_OTHER_NOTE = OTHER_NOTE_STRUCT.size

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


def rgb555_to_rgb888(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(
        (channel * 255 + COLOR_CHANNEL_MAX // 2) // COLOR_CHANNEL_MAX
        for channel in rgb
    )


def auto_other_notes(frames: list[dict]) -> list[dict]:
    """
    Note events from others_agreement_v4.json: each detected note lights
    AUTO_LEDS_PER_BAND LEDs for its frequency band, in its band colour.
    """

    notes = []

    for frame_index, detected in enumerate(calculate_other_notes(frames)):
        if not detected["note_detected"]:
            continue

        red, green, blue = rgb555_to_rgb888(detected["rgb"])
        first_led = detected["position"] * AUTO_LEDS_PER_BAND

        for position in range(first_led, first_led + AUTO_LEDS_PER_BAND):
            notes.append({
                "frame": frame_index,
                "position": position,
                "r": red,
                "g": green,
                "b": blue,
                "fade": AUTO_NOTE_FADE,
            })

    return notes


def manual_other_notes(data: dict, path: Path) -> list[dict]:
    """Note events from other_manual.json. Exits on invalid values."""

    notes = []

    for index, note in enumerate(data.get("notes", [])):
        try:
            frame = int(note["frame"])
            position = int(note["position"])
            values = [int(note[key]) for key in ("r", "g", "b", "fade")]
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"Error: note {index} in {path} is missing a field or is not a number.")

        if not 0 <= frame <= 0xFFFF:
            raise SystemExit(f"Error: note {index} in {path} has frame {frame}, outside 0-65535.")

        if not 0 <= position < OTHER_LED_COUNT:
            raise SystemExit(
                f"Error: note {index} in {path} has position {position}, "
                f"outside 0-{OTHER_LED_COUNT - 1}."
            )

        red, green, blue, fade = (int(clamp(value, 0, 255)) for value in values)

        notes.append({
            "frame": frame,
            "position": position,
            "r": red,
            "g": green,
            "b": blue,
            "fade": fade,
        })

    return notes


def pack_other_notes(notes: list[dict]) -> bytearray:
    """7 bytes per note, in the order given (sorted by frame, then LED)."""

    data = bytearray()

    for note in notes:
        data.extend(OTHER_NOTE_STRUCT.pack(
            note["frame"],
            note["position"],
            note["r"],
            note["g"],
            note["b"],
            note["fade"],
        ))

    return data


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
    vocal_frames: list[dict],
    other_notes: list[dict],
    other_source: dict,
) -> tuple[dict, bytearray, bytearray]:

    lengths = {
        "bass": len(bass_frames),
        "drums": len(drum_frames),
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
    vocal_frames = vocal_frames[:frame_count]

    drums = analyse_drums(drum_frames)

    # Other notes: keep those inside the song, one per frame and LED (the
    # later one wins), sorted by frame then LED so the ESP32 can read them
    # in order and binary-search them when seeking.
    by_key = {}
    outside = 0

    for note in other_notes:
        if note["frame"] >= frame_count:
            outside += 1
            continue

        by_key[(note["frame"], note["position"])] = note

    if outside:
        print(f"Warning: {outside} other notes start after the last frame and were skipped.")

    other_notes = [by_key[key] for key in sorted(by_key)]
    other_binary = pack_other_notes(other_notes)

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
        binary_data.extend(vocal_color_bits.to_bytes(2, "little"))
        binary_data.append(vocal_volume)
        binary_data.append(vocal_onset)

    output = {
        "format": {
            "byte_order": "little_endian",

            "frame_binary": {
                "bytes_per_frame": BYTES_PER_FRAME,
                "byte_0": "bass_volume",
                "byte_1": "drums",
                "bytes_2_3": "vocal_color_rgb555",
                "byte_4": "vocal_volume",
                "byte_5": "vocal_onset",
            },

            "drum_byte": {
                "bit_0": "kick_present",
                "bit_1": "hihat_present",
                "bits_2_4": "kick_intensity_0_to_7",
                "bits_5_7": "hihat_intensity_0_to_7",
            },

            "vocal_color": {
                "bits_0_4": "red_0_to_31",
                "bits_5_9": "green_0_to_31",
                "bits_10_14": "blue_0_to_31",
                "bit_15": "reserved",
            },

            "color_bands": {
                "red": RED_BANDS,
                "green": GREEN_BANDS,
                "blue": BLUE_BANDS,
            },

            "other_notes_binary": {
                "bytes_per_note": BYTES_PER_OTHER_NOTE,
                "order": "sorted by frame, then position",
                "bytes_0_1": "frame_uint16",
                "byte_2": f"position_0_to_{OTHER_LED_COUNT - 1}",
                "byte_3": "red_0_to_255",
                "byte_4": "green_0_to_255",
                "byte_5": "blue_0_to_255",
                "byte_6": "fade_0_to_255",
                "fade": (
                    "each frame after a note, brightness *= (255 - fade) / 255; "
                    "0 holds until replaced"
                ),
            },

            "other_source": other_source,
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

        "other_note_count": len(other_notes),

        "frames": readable_frames,

        "other_notes": other_notes,
    }

    return output, binary_data, other_binary


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

    parser.add_argument(
        "--other-binary-output",
        type=Path,
        default=DEFAULT_OTHER_BINARY_OUTPUT,
    )

    parser.add_argument(
        "--other",
        choices=("manual", "auto"),
        default=None,
        help=(
            f"Where 'other' notes come from: manual ({OTHER_MANUAL_FILENAME}) "
            f"or auto ({OTHER_INPUT_FILENAME}). Default: manual if the song "
            f"has it, otherwise auto."
        ),
    )

    return parser.parse_args()


def load_other_notes(song_directory: Path, choice: str | None) -> tuple[list[dict], dict]:
    """
    Load "other" notes from the chosen source, or by priority (manual, then
    auto) when none is chosen. Exits with an error if it can't.
    """

    manual_path = song_directory / OTHER_MANUAL_FILENAME
    auto_path = song_directory / OTHER_INPUT_FILENAME

    if choice is None:
        if manual_path.exists():
            choice = "manual"
        elif auto_path.exists():
            choice = "auto"
        else:
            raise SystemExit(
                f"Error: no 'other' notes for this song. Expected {manual_path} "
                f"(record with manual_other_recorder.py) or {auto_path} "
                f"(run other_agreement_compiler.py on other.mp3)."
            )

    path = manual_path if choice == "manual" else auto_path
    expected_format = OTHER_MANUAL_FORMAT if choice == "manual" else OTHER_EXPECTED_FORMAT

    if not path.exists():
        raise SystemExit(f"Error: {path} not found.")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    found_format = data.get("format") if isinstance(data, dict) else None

    if found_format != expected_format:
        raise SystemExit(
            f"Error: {path} has format {found_format!r}, "
            f"expected {expected_format!r}."
        )

    if choice == "manual":
        notes = manual_other_notes(data, path)
        details = {"fade_definition": data.get("fade_definition")}
    else:
        if not isinstance(data.get("frames"), list):
            raise SystemExit(f"Error: {path} must contain a 'frames' array.")

        notes = auto_other_notes(data["frames"])
        details = {
            "note_detection": data.get("note_detection", {}),
            "position_lookahead_frames": OTHER_LOOKAHEAD_FRAMES,
            "color_uses_lookahead_peak": OTHER_COLOR_USE_LOOKAHEAD_PEAK,
            "leds_per_band": AUTO_LEDS_PER_BAND,
            "fade": AUTO_NOTE_FADE,
        }

    print(f"Other notes:     {choice} ({path.name})")

    return notes, {"source": choice, "file": path.name, "format": expected_format, **details}


def main() -> None:

    args = parse_arguments()

    print("Loading input files...")

    song_directory = SONG_PATH / args.song

    bass_frames = load_frames(song_directory / "bass.json")
    drum_frames = load_frames(song_directory / "drums.json")

    vocal_frames = load_frames(song_directory / "vocals.json")

    other_notes, other_source = load_other_notes(song_directory, args.other)

    print("Processing visualisation data...")

    output, binary_data, other_binary = process(
        bass_frames,
        drum_frames,
        vocal_frames,
        other_notes,
        other_source,
    )

    json_output_path = song_directory / args.json_output
    binary_output_path = song_directory / args.binary_output
    other_binary_output_path = song_directory / args.other_binary_output

    json_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    binary_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    other_binary_output_path.parent.mkdir(
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

    with other_binary_output_path.open("wb") as file:
        file.write(other_binary)

    print()
    print(f"Frames:          {output['frame_count']}")
    print(f"Bytes/frame:     {BYTES_PER_FRAME}")
    print(f"Binary size:     {len(binary_data)} bytes")
    print(f"Other binary:    {len(other_binary)} bytes ({BYTES_PER_OTHER_NOTE} bytes/note)")

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
    print(f"JSON:         {json_output_path}")
    print(f"Binary:       {binary_output_path}")
    print(f"Other binary: {other_binary_output_path}")


if __name__ == "__main__":
    main()
