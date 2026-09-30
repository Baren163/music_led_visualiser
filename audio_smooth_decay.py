#!/usr/bin/env python3
"""
audio_smooth_decay.py

Post-process the JSON produced by audio_compile_16bands_40ms.py.

For every frequency band in every frame:

    decayed_value = previous_output_value - DECAY_PER_FRAME

Then:

    output_value = max(raw_input_value, decayed_value)

So the original JSON acts as the instantaneous "maximum/requested" value:
- If the new raw value is higher than the decayed previous value, the output
  immediately rises to the raw value.
- Otherwise, the output falls gradually toward zero.

This creates a peak-hold / falling-decay effect that is commonly useful for
audio visualizers.

Usage:
    python audio_smooth_decay.py input.json
    python audio_smooth_decay.py input.json output.json
    python audio_smooth_decay.py input.json output.json --decay 0.06

Default:
    DECAY_PER_FRAME = 0.05

At 25 frames per second (40 ms per frame), a value of 1.0 will take about
0.8 seconds to decay to zero with the default setting.
"""

import argparse
import json
from pathlib import Path


DEFAULT_DECAY_PER_FRAME = 0.05


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def smooth_frequency_data(data, decay_per_frame):
    frames = data.get("frames", [])

    if not frames:
        raise ValueError("Input JSON contains no frames.")

    first_frequencies = frames[0].get("frequencies", {})

    if not first_frequencies:
        raise ValueError("Input frames do not contain frequency data.")

    band_names = list(first_frequencies.keys())

    previous_values = {
        band: 0.0
        for band in band_names
    }

    output_frames = []

    for frame in frames:
        raw_frequencies = frame.get("frequencies", {})

        refined_frequencies = {}

        for band in band_names:
            raw_value = clamp01(
                raw_frequencies.get(band, 0.0)
            )

            decayed_value = max(
                0.0,
                previous_values[band] - decay_per_frame,
            )

            refined_value = max(
                raw_value,
                decayed_value,
            )

            refined_value = clamp01(
                refined_value
            )

            refined_frequencies[band] = round(
                refined_value,
                4,
            )

            previous_values[band] = refined_value

        # Preserve the rest of the original frame information.
        output_frame = dict(frame)
        output_frame["frequencies"] = refined_frequencies

        output_frames.append(
            output_frame
        )

    output_data = dict(data)

    output_data["format"] = "led_audio_features_smoothed_v1"
    output_data["processing"] = {
        "type": "decay_peak_hold",
        "decay_per_frame": decay_per_frame,
        "rule": "output = max(raw_value, previous_output - decay_per_frame)",
    }
    output_data["frames"] = output_frames

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Smooth audio frequency JSON using a falling decay "
            "with instantaneous upward response."
        )
    )

    parser.add_argument(
        "input",
        help="Input JSON produced by audio_compile_16bands_40ms.py",
    )

    parser.add_argument(
        "output",
        nargs="?",
        help=(
            "Output JSON filename. "
            "Default: input filename + _smoothed.json"
        ),
    )

    parser.add_argument(
        "--decay",
        type=float,
        default=DEFAULT_DECAY_PER_FRAME,
        help=(
            "Amount each frequency band decays per frame. "
            f"Default: {DEFAULT_DECAY_PER_FRAME}"
        ),
    )

    args = parser.parse_args()

    if args.decay < 0:
        raise SystemExit(
            "Decay must be zero or greater."
        )

    input_path = Path(
        args.input
    )

    if not input_path.exists():
        raise SystemExit(
            f"Input file not found: {input_path}"
        )

    if args.output:
        output_path = Path(
            args.output
        )
    else:
        output_path = input_path.with_name(
            input_path.stem + "_smoothed.json"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(
            f
        )

    output_data = smooth_frequency_data(
        data,
        args.decay,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_data,
            f,
            indent=2,
        )

    frame_duration = float(
        output_data.get(
            "frame_duration",
            0.04,
        )
    )

    if args.decay > 0:
        decay_seconds = (
            1.0 / args.decay
        ) * frame_duration
    else:
        decay_seconds = float("inf")

    print(
        f"Wrote: {output_path}"
    )
    print(
        f"Frames: {len(output_data['frames'])}"
    )
    print(
        f"Decay per frame: {args.decay}"
    )

    if decay_seconds != float("inf"):
        print(
            "Approx. time for a value of 1.0 "
            f"to decay to zero: {decay_seconds:.2f} seconds"
        )
    else:
        print(
            "Decay is 0, so values will never fall."
        )


if __name__ == "__main__":
    main()
