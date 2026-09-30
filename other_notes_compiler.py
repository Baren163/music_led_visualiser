#!/usr/bin/env python3
"""
Detect notes in the Demucs "other" stem and group them by sound type.

Pipeline:

    1. basic-pitch transcribes the stem into note events
       (start, end, MIDI pitch, amplitude).
    2. Quiet and very short notes are removed.
    3. Notes that start together are merged into one chord event.
    4. Events that are too close together are thinned out, keeping the
       loudest, so the strip never gets busier than MIN_EVENT_GAP allows.
    5. Each event gets a timbre fingerprint (MFCCs, harmonic profile,
       loudness envelope) and events are clustered into sound types.
       Each sound type becomes one colour index.
    6. LED position is the event's average pitch, mapped onto the song's
       pitch range.

Output is a 25 Hz frame JSON (format "led_audio_notes_v1") that
generate_song_vis.py reads in place of other_onsets.json.

Usage:

    python other_notes_compiler.py separated/htdemucs/timeless/other.mp3

Optional:

    python other_notes_compiler.py other.mp3 --colors 5 --min-gap 0.15
"""

import argparse
import json
import warnings
from pathlib import Path

import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FRAME_RATE = 25.0
FRAME_DURATION = 1.0 / FRAME_RATE

# Sample rate used for timbre analysis.
ANALYSIS_SAMPLE_RATE = 22050


# -----------------------------------------------------------------------------
# Note detection
# -----------------------------------------------------------------------------

# Notes quieter than this (basic-pitch amplitude, ~0.3-0.6 typical) are dropped.
MIN_NOTE_AMPLITUDE = 0.35

# Notes shorter than this are dropped.
MIN_NOTE_DURATION = 0.10

# Notes starting within this many seconds of each other form one chord event.
CHORD_WINDOW = 0.05

# Minimum time between two displayed events. When events are closer than
# this, only the loudest is kept. Larger = calmer strip.
MIN_EVENT_GAP = 0.12


# -----------------------------------------------------------------------------
# Position
# -----------------------------------------------------------------------------

POSITION_COUNT = 14

# The pitch range mapped onto the strip is taken from these percentiles of
# the song's event pitches, so rare outliers don't squash everything else.
POSITION_PITCH_PERCENTILES = (5.0, 95.0)


# -----------------------------------------------------------------------------
# Sound-type clustering
# -----------------------------------------------------------------------------

# Colour index must fit in 3 bits.
MAX_COLORS = 8

# Range of cluster counts tried when --colors is "auto".
AUTO_COLOR_RANGE = (3, 6)

# Audio after each event start used for the spectral fingerprint.
TIMBRE_WINDOW = 0.15

# Audio after each event start used for the loudness-envelope fingerprint.
ENVELOPE_WINDOW = 0.30
ENVELOPE_SEGMENTS = 6

HARMONIC_COUNT = 8
MFCC_COUNT = 13

RANDOM_SEED = 0


def load_note_events(input_path):
    # basic-pitch pulls in TensorFlow, so import it only when needed.
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, note_events = predict(
            str(input_path),
            ICASSP_2022_MODEL_PATH,
        )

    return [
        {
            "start": float(start),
            "end": float(end),
            "pitch": int(pitch),
            "amplitude": float(amplitude),
        }
        for start, end, pitch, amplitude, _ in note_events
    ]


def filter_notes(notes, min_amplitude, min_duration):
    return [
        note
        for note in notes
        if note["amplitude"] >= min_amplitude
        and note["end"] - note["start"] >= min_duration
    ]


def group_chords(notes, chord_window):
    notes = sorted(notes, key=lambda note: note["start"])

    groups = []

    for note in notes:
        if groups and note["start"] - groups[-1][0]["start"] <= chord_window:
            groups[-1].append(note)
        else:
            groups.append([note])

    events = []

    for group in groups:
        amplitudes = np.array([note["amplitude"] for note in group])
        pitches = np.array([note["pitch"] for note in group])
        lead = group[int(np.argmax(amplitudes))]

        events.append(
            {
                "start": group[0]["start"],
                "end": max(note["end"] for note in group),
                "pitch": float(np.average(pitches, weights=amplitudes)),
                "lead_pitch": lead["pitch"],
                "amplitude": float(np.max(amplitudes)),
                "note_count": len(group),
            }
        )

    return events


def thin_events(events, min_gap):
    # Greedy non-maximum suppression: accept the loudest events first and
    # reject anything within min_gap of an already accepted event.
    accepted = []

    for event in sorted(events, key=lambda e: e["amplitude"], reverse=True):
        if all(
            abs(event["start"] - other["start"]) >= min_gap
            for other in accepted
        ):
            accepted.append(event)

    return sorted(accepted, key=lambda e: e["start"])


def segment(y, sample_rate, start_seconds, length_seconds):
    start = int(start_seconds * sample_rate)
    length = int(length_seconds * sample_rate)

    seg = y[start:start + length]

    if len(seg) < length:
        seg = np.pad(seg, (0, length - len(seg)))

    return seg


def harmonic_profile(seg, sample_rate, midi_pitch):
    # Level of each harmonic relative to the fundamental, in dB. This
    # describes timbre independently of which note is played.
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    frequencies = np.fft.rfftfreq(len(seg), 1.0 / sample_rate)

    f0 = librosa.midi_to_hz(midi_pitch)
    levels = []

    for harmonic in range(1, HARMONIC_COUNT + 1):
        target = f0 * harmonic
        mask = np.abs(frequencies - target) <= target * 0.03

        if target >= sample_rate / 2 or not np.any(mask):
            levels.append(0.0)
        else:
            levels.append(float(np.max(spectrum[mask])))

    levels = np.asarray(levels)
    db = 20.0 * np.log10(levels / max(levels[0], 1e-9) + 1e-9)

    return np.clip(db, -60.0, 20.0)


def envelope_shape(seg):
    # Loudness over time after the onset, normalised to its peak, so
    # plucked, sustained and swelling sounds look different.
    chunks = np.array_split(seg, ENVELOPE_SEGMENTS)
    rms = np.array([np.sqrt(np.mean(chunk ** 2)) for chunk in chunks])

    return rms / max(np.max(rms), 1e-9)


def timbre_features(y, sample_rate, events):
    mfcc_rows = []
    harmonic_rows = []
    envelope_rows = []
    flatness_rows = []

    for event in events:
        timbre_seg = segment(y, sample_rate, event["start"], TIMBRE_WINDOW)
        envelope_seg = segment(y, sample_rate, event["start"], ENVELOPE_WINDOW)

        mfcc = librosa.feature.mfcc(
            y=timbre_seg,
            sr=sample_rate,
            n_mfcc=MFCC_COUNT + 1,
            n_fft=2048,
            hop_length=512,
        )

        # Coefficient 0 is overall loudness, which is not a timbre cue.
        mfcc_rows.append(np.mean(mfcc[1:], axis=1))

        harmonic_rows.append(
            harmonic_profile(timbre_seg, sample_rate, event["lead_pitch"])
        )

        envelope_rows.append(envelope_shape(envelope_seg))

        flatness_rows.append(
            [float(np.mean(librosa.feature.spectral_flatness(y=timbre_seg)))]
        )

    groups = [
        np.asarray(mfcc_rows),
        np.asarray(harmonic_rows),
        np.asarray(envelope_rows),
        np.asarray(flatness_rows),
    ]

    # Standardise every column, then weight each group equally regardless
    # of how many columns it has.
    weighted = []

    for group in groups:
        scaled = StandardScaler().fit_transform(group)
        weighted.append(scaled / np.sqrt(group.shape[1]))

    return np.hstack(weighted)


def cluster_sound_types(features, color_setting):
    if len(features) < 2:
        return np.zeros(len(features), dtype=int), 1, {}

    if color_setting == "auto":
        low, high = AUTO_COLOR_RANGE
        candidates = range(low, min(high, len(features) - 1) + 1)
    else:
        candidates = [min(int(color_setting), len(features))]

    best_labels = None
    best_score = -np.inf
    scores = {}

    for k in candidates:
        labels = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_SEED,
        ).fit_predict(features)

        if k > 1:
            score = silhouette_score(features, labels, random_state=RANDOM_SEED)
        else:
            score = 0.0

        scores[k] = round(float(score), 4)

        if best_labels is None or score > best_score:
            best_labels = labels
            best_score = score

    # Renumber so colour 0 is the most common sound type, 1 the next, etc.
    counts = np.bincount(best_labels)
    order = np.argsort(-counts)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))

    return remap[best_labels], len(order), scores


def assign_positions(events):
    pitches = np.array([event["pitch"] for event in events])

    low, high = np.percentile(pitches, POSITION_PITCH_PERCENTILES)
    span = max(high - low, 1e-9)

    positions = np.round(
        (pitches - low) / span * (POSITION_COUNT - 1)
    )

    return np.clip(positions, 0, POSITION_COUNT - 1).astype(int), low, high


def compile_notes(
    input_path,
    output_path,
    color_setting,
    min_gap,
    min_amplitude,
):
    print(f"Loading: {input_path}")

    y, sample_rate = librosa.load(
        input_path,
        sr=ANALYSIS_SAMPLE_RATE,
        mono=True,
    )

    duration = len(y) / sample_rate

    # Match the frame count of the other analysis files (center=True STFT
    # at 25 Hz gives one frame per hop plus one).
    frame_count = int(np.floor(duration * FRAME_RATE)) + 1

    print("Transcribing notes with basic-pitch...")

    notes = load_note_events(input_path)
    raw_note_count = len(notes)

    notes = filter_notes(notes, min_amplitude, MIN_NOTE_DURATION)
    events = group_chords(notes, CHORD_WINDOW)
    chord_event_count = len(events)
    events = thin_events(events, min_gap)

    if not events:
        raise SystemExit("No notes survived filtering. Try a lower --min-amplitude.")

    print("Clustering sound types...")

    features = timbre_features(y, sample_rate, events)
    colors, color_count, scores = cluster_sound_types(features, color_setting)
    positions, pitch_low, pitch_high = assign_positions(events)

    amplitudes = np.array([event["amplitude"] for event in events])
    amplitude_reference = max(np.percentile(amplitudes, 95.0), 1e-9)

    frames = [
        {
            "time": round(i * FRAME_DURATION, 5),
            "note_detected": False,
            "brightness": 0.0,
            "position": 0,
            "color": 0,
        }
        for i in range(frame_count)
    ]

    for event, position, color in zip(events, positions, colors):
        event["position"] = int(position)
        event["color"] = int(color)
        event["brightness"] = float(
            np.clip(event["amplitude"] / amplitude_reference, 0.0, 1.0)
        )

        frame_index = int(round(event["start"] * FRAME_RATE))

        if not 0 <= frame_index < frame_count:
            continue

        frame = frames[frame_index]

        # min_gap is normally longer than a frame, but keep the louder event
        # if two ever land on the same frame.
        if frame["note_detected"] and frame["brightness"] >= event["brightness"]:
            continue

        frame["note_detected"] = True
        frame["brightness"] = round(event["brightness"], 4)
        frame["position"] = event["position"]
        frame["color"] = event["color"]

    color_summary = []

    for color in range(color_count):
        members = [event for event in events if event["color"] == color]

        color_summary.append(
            {
                "color": color,
                "event_count": len(members),
                "mean_pitch": round(
                    float(np.mean([event["pitch"] for event in members])),
                    2,
                ),
            }
        )

    output = {
        "format": "led_audio_notes_v1",
        "source": input_path.name,
        "duration": round(float(duration), 5),
        "frame_rate": FRAME_RATE,
        "frame_duration": FRAME_DURATION,
        "note_detection": {
            "method": "basic_pitch",
            "min_note_amplitude": min_amplitude,
            "min_note_duration": MIN_NOTE_DURATION,
            "chord_window": CHORD_WINDOW,
            "min_event_gap": min_gap,
            "raw_note_count": raw_note_count,
            "chord_event_count": chord_event_count,
            "event_count": len(events),
        },
        "position": {
            "position_count": POSITION_COUNT,
            "method": "amplitude-weighted average pitch",
            "pitch_range_midi": [
                round(float(pitch_low), 2),
                round(float(pitch_high), 2),
            ],
        },
        "sound_types": {
            "method": "kmeans on MFCC + harmonic profile + envelope + flatness",
            "color_count": color_count,
            "silhouette_scores": scores,
            "colors": color_summary,
        },
        "events": [
            {
                "start": round(event["start"], 4),
                "end": round(event["end"], 4),
                "pitch": round(event["pitch"], 2),
                "note_count": event["note_count"],
                "brightness": round(event["brightness"], 4),
                "position": event["position"],
                "color": event["color"],
            }
            for event in events
        ],
        "frames": frames,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Wrote: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Raw notes: {raw_note_count}")
    print(f"Chord events after filtering: {chord_event_count}")
    print(
        f"Displayed events: {len(events)} "
        f"({len(events) / duration:.2f} per second)"
    )
    print(f"Pitch range mapped to strip: MIDI {pitch_low:.1f}-{pitch_high:.1f}")

    if scores:
        print(f"Silhouette scores by colour count: {scores}")

    print(f"Sound types: {color_count}")

    for entry in color_summary:
        print(
            f"  colour {entry['color']}: "
            f"{entry['event_count']:4d} events, "
            f"mean pitch {entry['mean_pitch']:.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe the 'other' stem into 25 Hz note frames with "
            "pitch-based position and sound-type colour."
        )
    )

    parser.add_argument(
        "input",
        help="Input audio file (normally the Demucs other.mp3).",
    )

    parser.add_argument(
        "output",
        nargs="?",
        help="Output JSON. Default: <input>_notes.json",
    )

    parser.add_argument(
        "--colors",
        default="auto",
        help=(
            f"Number of sound types (1-{MAX_COLORS}), or 'auto' to pick "
            f"{AUTO_COLOR_RANGE[0]}-{AUTO_COLOR_RANGE[1]} by silhouette score. "
            "Default: auto"
        ),
    )

    parser.add_argument(
        "--min-gap",
        type=float,
        default=MIN_EVENT_GAP,
        help=f"Minimum seconds between displayed events. Default: {MIN_EVENT_GAP}",
    )

    parser.add_argument(
        "--min-amplitude",
        type=float,
        default=MIN_NOTE_AMPLITUDE,
        help=f"Drop notes quieter than this. Default: {MIN_NOTE_AMPLITUDE}",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_name(input_path.stem + "_notes.json")

    if args.colors != "auto":
        try:
            colors = int(args.colors)
        except ValueError:
            raise SystemExit("--colors must be a number or 'auto'.")

        if not 1 <= colors <= MAX_COLORS:
            raise SystemExit(f"--colors must be between 1 and {MAX_COLORS}.")

    if args.min_gap < 0.0:
        raise SystemExit("--min-gap must be zero or greater.")

    compile_notes(
        input_path=input_path,
        output_path=output_path,
        color_setting=args.colors,
        min_gap=args.min_gap,
        min_amplitude=args.min_amplitude,
    )


if __name__ == "__main__":
    main()
