#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import demucs.separate

DEFAULT_MODEL = "htdemucs_ft"
DEFAULT_BITRATE = 320


def separate_track(input_path, output_dir, model, bitrate, device):
    args = [
        "--mp3",
        "--mp3-bitrate", str(bitrate),
        "-n", model,
        "-o", str(output_dir),
        "--shifts", "2"
    ]

    if device:
        args.extend(["-d", device])

    args.append(str(input_path))

    print("Demucs 4-stem separator")
    print("-----------------------")
    print(f"Input:   {input_path}")
    print(f"Model:   {model}")
    print(f"Bitrate: {bitrate} kbps")
    print(f"Output:  {output_dir}")
    print(f"Device:  {device or 'automatic'}")
    print()
    print("Separating...")
    print()

    demucs.separate.main(args)

    expected_folder = output_dir / model / input_path.stem

    print()
    print("Finished.")
    print(f"Expected stem folder: {expected_folder}")
    print()
    print("Expected outputs:")

    for name in ("vocals.mp3", "drums.mp3", "bass.mp3", "other.mp3"):
        stem = expected_folder / name
        print(("  OK  " if stem.exists() else "  ?   ") + str(stem))


def main():
    parser = argparse.ArgumentParser(
        description="Split an MP3 into vocals, drums, bass, and other using Demucs."
    )
    parser.add_argument("input", help="Input MP3 file.")
    parser.add_argument("--output", default="separated", help="Base output directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Demucs model name.")
    parser.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE, help="Output MP3 bitrate in kbps.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None, help="Force processing device.")

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        separate_track(
            input_path=input_path,
            output_dir=output_dir,
            model=args.model,
            bitrate=args.bitrate,
            device=args.device,
        )
    except KeyboardInterrupt:
        print("\nSeparation cancelled.")
        raise SystemExit(130)
    except Exception as exc:
        print("\nDemucs separation failed.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
