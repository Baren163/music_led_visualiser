#!/usr/bin/env python3
import argparse
import bisect
import json
import time
from pathlib import Path

import pygame

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
TARGET_FPS = 120
SEEK_SECONDS = 5.0
TOP_MARGIN = 90
BOTTOM_MARGIN = 150
SIDE_MARGIN = 70

ONSETS_V1_FORMAT = "led_audio_onsets_v1"
AGREEMENT_V4_FORMAT = "others_agreement_v4"
AGREEMENT_DEBUG_FORMAT = "others_agreement_debug_v4_1"

# Formats that show the note-detection circle, and its lit colour.
NOTE_INDICATOR_COLOURS = {
    ONSETS_V1_FORMAT: (255, 230, 120),        # yellow
    AGREEMENT_V4_FORMAT: (255, 105, 200),     # pink
    AGREEMENT_DEBUG_FORMAT: (255, 105, 200),  # pink
}

# The debug view has an extra agreement meter, so the chart sits higher.
DEBUG_EXTRA_BOTTOM_MARGIN = 30


def load_audio_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        raise ValueError("The JSON file contains no frames.")

    format_string = str(data.get("format", ""))

    timestamps = [
        float(frame.get("time", 0.0))
        for frame in frames
    ]

    if any(b < a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Frame timestamps must be ascending.")

    # --------------------------------------------------------
    # Choose which per-band data to display.
    #
    # Existing frequency compiler:
    #   frame["frequencies"]
    #
    # New onset compiler:
    #   frame["onset_bands"]
    #
    # Agreement debug file:
    #   frame["frequency_changes"]
    # --------------------------------------------------------
    if format_string == ONSETS_V1_FORMAT:
        band_field = "onset_bands"
        display_mode = "onset_bands"
    elif format_string == AGREEMENT_DEBUG_FORMAT:
        band_field = "frequency_changes"
        display_mode = "frequency_changes"
    elif "frequencies" in frames[0]:
        band_field = "frequencies"
        display_mode = "frequencies"
    elif "onset_bands" in frames[0]:
        # Fallback for onset-style files with a missing/unknown format string.
        band_field = "onset_bands"
        display_mode = "onset_bands"
    else:
        raise ValueError(
            "Unsupported JSON frame format: expected either "
            "'frequencies' or 'onset_bands' in each frame."
        )

    first_band_data = frames[0].get(band_field, {})
    if not isinstance(first_band_data, dict) or not first_band_data:
        raise ValueError(
            f"The first frame contains no usable '{band_field}' data."
        )

    band_names = list(first_band_data.keys())

    magnitudes = []

    for frame in frames:
        band_data = frame.get(band_field, {})

        frame_values = [
            max(
                0.0,
                min(
                    1.0,
                    float(band_data.get(band, 0.0)),
                ),
            )
            for band in band_names
        ]

        magnitudes.append(frame_values)

    # Older frequency files may contain volume.
    # New onset-only files do not, so missing values become zero.
    # The agreement debug file shows the volume change instead.
    volume_field = (
        "volume_change"
        if display_mode == "frequency_changes"
        else "volume"
    )

    volumes = [
        max(
            0.0,
            min(
                1.0,
                float(frame.get(volume_field, 0.0)),
            ),
        )
        for frame in frames
    ]

    # Agreement value and threshold (agreement debug file only).
    agreements = [
        max(
            0.0,
            min(
                1.0,
                float(frame.get("agreement", 0.0)),
            ),
        )
        for frame in frames
    ]

    agreement_threshold = float(
        data.get("note_detection", {}).get("agreement_threshold", 0.0)
    )

    # Read overall onset when present.
    # Older files without onset safely fall back to zero.
    onsets = [
        max(
            0.0,
            min(
                1.0,
                float(frame.get("onset", 0.0)),
            ),
        )
        for frame in frames
    ]

    # Read note detection when present.
    note_detected = [
        bool(frame.get("note_detected", False))
        for frame in frames
    ]

    duration = max(
        float(data.get("duration", timestamps[-1])),
        timestamps[-1],
    )

    frame_duration = float(
        data.get(
            "frame_duration",
            1.0 / float(data.get("frame_rate", 25.0)),
        )
    )

    return (
        timestamps,
        band_names,
        magnitudes,
        volumes,
        onsets,
        note_detected,
        agreements,
        agreement_threshold,
        duration,
        frame_duration,
        format_string,
        display_mode,
    )


def frame_index_for_time(timestamps, current_time):
    if current_time <= timestamps[0]:
        return 0
    if current_time >= timestamps[-1]:
        return len(timestamps) - 1

    pos = bisect.bisect_left(timestamps, current_time)
    before = pos - 1
    after = pos

    if current_time - timestamps[before] <= timestamps[after] - current_time:
        return before
    return after


def format_time(seconds):
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Real-time viewer for audio frequency JSON."
    )
    parser.add_argument("input", help="JSON file produced by audio_compile.py")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    (
        timestamps,
        band_names,
        magnitudes,
        volumes,
        onsets,
        note_detected,
        agreements,
        agreement_threshold,
        duration,
        frame_duration,
        format_string,
        display_mode,
    ) = load_audio_data(input_path)

    is_debug = display_mode == "frequency_changes"

    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.RESIZABLE,
    )
    pygame.display.set_caption(f"Audio Frequency Viewer - {input_path.name}")

    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont(None, 34)
    label_font = pygame.font.SysFont(None, 24)
    small_font = pygame.font.SysFont(None, 20)

    background = (18, 18, 22)
    foreground = (235, 235, 240)
    muted = (150, 150, 160)
    grid_colour = (55, 55, 65)
    frequency_bar_colour = (110, 180, 240)
    onset_bar_colour = (225, 205, 105)  # mild yellow
    change_bar_colour = (120, 215, 165)  # soft green
    agreement_colour = (255, 105, 200)  # pink
    volume_colour = (190, 190, 210)

    if display_mode == "onset_bands":
        bar_colour = onset_bar_colour
    elif display_mode == "frequency_changes":
        bar_colour = change_bar_colour
    else:
        bar_colour = frequency_bar_colour

    running = True
    paused = False
    playback_offset = 0.0
    playback_started_at = time.perf_counter()

    while running:
        now = time.perf_counter()
        playback_time = playback_offset if paused else playback_offset + (now - playback_started_at)

        if playback_time >= duration:
            playback_time = duration
            playback_offset = duration
            paused = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    if paused:
                        playback_started_at = time.perf_counter()
                        paused = False
                    else:
                        playback_offset = playback_time
                        paused = True

                elif event.key == pygame.K_LEFT:
                    playback_offset = max(0.0, playback_time - SEEK_SECONDS)
                    playback_started_at = time.perf_counter()

                elif event.key == pygame.K_RIGHT:
                    playback_offset = min(duration, playback_time + SEEK_SECONDS)
                    playback_started_at = time.perf_counter()

                elif event.key == pygame.K_HOME:
                    playback_offset = 0.0
                    playback_started_at = time.perf_counter()

                elif event.key == pygame.K_END:
                    playback_offset = duration
                    playback_started_at = time.perf_counter()
                    paused = True

        now = time.perf_counter()
        playback_time = playback_offset if paused else playback_offset + (now - playback_started_at)
        playback_time = min(playback_time, duration)

        idx = frame_index_for_time(timestamps, playback_time)
        values = magnitudes[idx]
        volume = volumes[idx]
        onset = onsets[idx]
        detected = note_detected[idx]
        agreement = agreements[idx]

        width, height = screen.get_size()
        screen.fill(background)

        if display_mode == "onset_bands":
            title_text = "Audio Frequency-Band Onset Spectrum"
        elif display_mode == "frequency_changes":
            title_text = "Note Agreement Inputs - Frequency-Band Increase"
        else:
            title_text = "Audio Frequency Spectrum"

        title = title_font.render(title_text, True, foreground)
        screen.blit(title, (SIDE_MARGIN, 30))

        status = label_font.render("PAUSED" if paused else "PLAYING", True, foreground)
        screen.blit(status, (width - SIDE_MARGIN - status.get_width(), 35))

        chart_left = SIDE_MARGIN
        chart_right = width - SIDE_MARGIN
        chart_top = TOP_MARGIN
        chart_bottom = height - BOTTOM_MARGIN

        if is_debug:
            chart_bottom -= DEBUG_EXTRA_BOTTOM_MARGIN

        chart_width = max(1, chart_right - chart_left)
        chart_height = max(1, chart_bottom - chart_top)

        for step in range(5):
            magnitude = step / 4
            y = chart_bottom - int(magnitude * chart_height)
            pygame.draw.line(screen, grid_colour, (chart_left, y), (chart_right, y), 1)
            label = small_font.render(f"{magnitude:.2f}", True, muted)
            screen.blit(label, (10, y - 8))

        band_count = len(band_names)
        slot_width = chart_width / band_count
        bar_width = max(8, int(slot_width * 0.45))

        for i, (name, magnitude) in enumerate(zip(band_names, values)):
            centre_x = chart_left + int((i + 0.5) * slot_width)
            bar_height = int(magnitude * chart_height)

            rect = pygame.Rect(
                centre_x - bar_width // 2,
                chart_bottom - bar_height,
                bar_width,
                bar_height,
            )
            pygame.draw.rect(
                screen,
                bar_colour,
                rect,
                border_radius=max(1, bar_width // 5),
            )

            label = small_font.render(name, True, foreground)
            screen.blit(
                label,
                (centre_x - label.get_width() // 2, chart_bottom + 14),
            )

        meter_left = SIDE_MARGIN + 90
        meter_right = width - SIDE_MARGIN - 35
        meter_width = max(1, meter_right - meter_left)

        volume_y = height - 92
        onset_y = height - 62
        agreement_y = height - 122

        if is_debug:
            agreement_label = label_font.render("Agree", True, foreground)
            screen.blit(agreement_label, (SIDE_MARGIN, agreement_y - 8))

            pygame.draw.rect(
                screen,
                grid_colour,
                pygame.Rect(meter_left, agreement_y, meter_width, 16),
                border_radius=4,
            )
            pygame.draw.rect(
                screen,
                agreement_colour,
                pygame.Rect(
                    meter_left,
                    agreement_y,
                    int(meter_width * agreement),
                    16,
                ),
                border_radius=4,
            )

            # Threshold marker: a note needs agreement past this line.
            threshold_x = meter_left + int(meter_width * agreement_threshold)
            pygame.draw.line(
                screen,
                foreground,
                (threshold_x, agreement_y - 4),
                (threshold_x, agreement_y + 19),
                2,
            )

        volume_label = label_font.render(
            "Vol rise" if is_debug else "Volume",
            True,
            foreground,
        )
        screen.blit(volume_label, (SIDE_MARGIN, volume_y - 8))

        pygame.draw.rect(
            screen,
            grid_colour,
            pygame.Rect(meter_left, volume_y, meter_width, 16),
            border_radius=4,
        )
        pygame.draw.rect(
            screen,
            volume_colour,
            pygame.Rect(
                meter_left,
                volume_y,
                int(meter_width * volume),
                16,
            ),
            border_radius=4,
        )

        onset_label = label_font.render("Onset", True, foreground)
        screen.blit(onset_label, (SIDE_MARGIN, onset_y - 8))

        pygame.draw.rect(
            screen,
            grid_colour,
            pygame.Rect(meter_left, onset_y, meter_width, 16),
            border_radius=4,
        )
        pygame.draw.rect(
            screen,
            onset_bar_colour,
            pygame.Rect(
                meter_left,
                onset_y,
                int(meter_width * onset),
                16,
            ),
            border_radius=4,
        )

        # For formats with note detection, show the note-detection state.
        if format_string in NOTE_INDICATOR_COLOURS:
            indicator_x = width - SIDE_MARGIN + 22
            indicator_y = onset_y + 8

            indicator_colour = (
                NOTE_INDICATOR_COLOURS[format_string]
                if detected
                else (55, 55, 65)
            )

            pygame.draw.circle(
                screen,
                indicator_colour,
                (indicator_x, indicator_y),
                9,
            )

        timeline_y = height - 28
        timeline_left = SIDE_MARGIN
        timeline_right = width - SIDE_MARGIN
        timeline_width = max(1, timeline_right - timeline_left)

        pygame.draw.line(
            screen,
            grid_colour,
            (timeline_left, timeline_y),
            (timeline_right, timeline_y),
            4,
        )

        progress = playback_time / duration if duration > 0 else 0.0
        playhead_x = timeline_left + int(progress * timeline_width)
        pygame.draw.circle(screen, foreground, (playhead_x, timeline_y), 7)

        time_surface = small_font.render(
            f"{format_time(playback_time)} / {format_time(duration)}",
            True,
            foreground,
        )
        screen.blit(
            time_surface,
            (width // 2 - time_surface.get_width() // 2, height - 55),
        )

        controls_surface = small_font.render(
            "Space: pause/resume   Left/Right: seek 5 s   Home/End: start/end   Esc: quit",
            True,
            muted,
        )
        screen.blit(
            controls_surface,
            (width // 2 - controls_surface.get_width() // 2, 66),
        )

        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
