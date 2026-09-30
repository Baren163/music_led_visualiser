#!/usr/bin/env python3
"""
Real-time 144-LED strip visualizer for led_visualisation_data.json.

Usage:
    pip install pygame
    python led_strip_visualizer.py path/to/led_visualisation_data.json

Controls:
    SPACE       Pause / resume
    LEFT        Seek backward 1 second
    RIGHT       Seek forward 1 second
    R           Restart
    ESC         Quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pygame


# =============================================================================
# CONFIGURATION
# =============================================================================

FPS = 25
LED_COUNT = 144
HALF_LED_COUNT = 72

WINDOW_WIDTH = 1450
WINDOW_HEIGHT = 600
BACKGROUND = (8, 8, 8)

BASS_COLOR = (255, 0, 0)
DRUM_COLOR = (255, 70, 0)
OTHER_COLOR = (255, 125, 0)
VOCAL_COLOR = (255, 225, 0)

LED_RADIUS = 5
GLOW_ENABLED = True
GLOW_RADIUS_MULTIPLIER = 3.2
BRIGHTNESS_GAMMA = 0.85

KICK_FADE_PER_FRAME = 0.82
HIHAT_FADE_PER_FRAME = 0.74

# Drums occupy 18 LEDs total: 9 kick + 9 hi-hat.
DRUM_LEDS_PER_TYPE = 9

# Brightness multiplier across each 9-LED drum section.
# The middle three LEDs are full brightness. Moving outward, the final
# three LEDs on each side fade down to black.
DRUM_BRIGHTNESS_PROFILE = [
    0.0,
    0.25,
    0.55,
    1.0,
    1.0,
    1.0,
    0.55,
    0.25,
    0.0,
]

# Other note/event fade. The new 'other' representation contains one
# note position rather than an 8-band spectrum, so each LED in the
# 18-LED region keeps its own fading brightness state.
OTHER_FADE_PER_FRAME = 0.78

# A detected note lights its selected LED at full event brightness and
# its immediate neighbour on each side at this fraction of that level.
OTHER_ADJACENT_BRIGHTNESS = 0.25

# There are 14 possible note positions from the 16-band, 3-band rolling
# window. They are mapped to 14 consecutive centre LEDs inside the
# 18-LED Other region, leaving two LEDs of margin at each edge.
OTHER_POSITION_COUNT = 14
OTHER_POSITION_START_OFFSET = 2

SPECTRUM_LED_COUNTS = [3, 2, 2, 2, 2, 2, 2, 3]

FREQUENCY_BANDS = [
    "40-80Hz",
    "80-159Hz",
    "159-317Hz",
    "317-632Hz",
    "632-1261Hz",
    "1261-2515Hz",
    "2515-5015Hz",
    "5015-10000Hz",
]

# =============================================================================
# END CONFIGURATION
# =============================================================================


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_frames(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict) and isinstance(data.get("frames"), list):
        frames = data["frames"]
    else:
        raise ValueError(
            "JSON must be either a list of frames or an object containing a 'frames' list."
        )

    if not frames:
        raise ValueError("The JSON contains no frames.")

    return frames


def scale_color(base_color: tuple[int, int, int], brightness: float) -> tuple[int, int, int]:
    brightness = clamp(brightness, 0.0, 1.0)
    brightness = brightness ** BRIGHTNESS_GAMMA
    return tuple(round(channel * brightness) for channel in base_color)


def get_spectrum_values(frame: dict, component: str) -> list[int]:
    spectrum = frame.get(component, {})
    return [
        int(clamp(float(spectrum.get(band, 0)), 0, 255))
        for band in FREQUENCY_BANDS
    ]


def render_spectrum_region(
    leds: list[tuple[tuple[int, int, int], float]],
    start: int,
    spectrum: list[int],
    base_color: tuple[int, int, int],
) -> None:
    led_index = start

    for value, count in zip(spectrum, SPECTRUM_LED_COUNTS):
        brightness = value / 255.0
        for _ in range(count):
            leds[led_index] = (base_color, brightness)
            led_index += 1


class LEDRenderer:
    def __init__(self) -> None:
        self.kick_level = 0.0
        self.hihat_level = 0.0
        self.other_levels = [0.0] * 18

    def reset(self) -> None:
        self.kick_level = 0.0
        self.hihat_level = 0.0
        self.other_levels = [0.0] * 18

    def render_frame(
        self,
        frame: dict,
    ) -> list[tuple[tuple[int, int, int], float]]:

        leds = [((0, 0, 0), 0.0) for _ in range(LED_COUNT)]

        # Bass: LEDs 0-17
        bass_byte = int(
            clamp(float(frame.get("bass", {}).get("byte", 0)), 0, 255)
        )

        bass_position = (bass_byte / 255.0) * 18.0

        for i in range(18):
            brightness = clamp(bass_position - i, 0.0, 1.0)
            leds[i] = (BASS_COLOR, brightness)

        # Drums: LEDs 18-26 kick, 27-35 hi-hat.
        #
        # Both drum types receive 9 LEDs. The detected intensity sets the
        # overall level, while DRUM_BRIGHTNESS_PROFILE shapes the section so
        # the centre is bright and the edges fade smoothly to black.
        drums = frame.get("drums", {})

        if drums.get("kick", False):
            intensity = int(clamp(float(drums.get("kick_intensity", 0)), 0, 7))
            self.kick_level = max(self.kick_level, intensity / 7.0)
        else:
            self.kick_level *= KICK_FADE_PER_FRAME

        if drums.get("hihat", False):
            intensity = int(clamp(float(drums.get("hihat_intensity", 0)), 0, 7))
            self.hihat_level = max(self.hihat_level, intensity / 7.0)
        else:
            self.hihat_level *= HIHAT_FADE_PER_FRAME

        if self.kick_level < 0.001:
            self.kick_level = 0.0
        if self.hihat_level < 0.001:
            self.hihat_level = 0.0

        if len(DRUM_BRIGHTNESS_PROFILE) != DRUM_LEDS_PER_TYPE:
            raise ValueError(
                "DRUM_BRIGHTNESS_PROFILE must contain exactly "
                f"{DRUM_LEDS_PER_TYPE} values."
            )

        for offset, multiplier in enumerate(DRUM_BRIGHTNESS_PROFILE):
            leds[18 + offset] = (
                DRUM_COLOR,
                self.kick_level * multiplier,
            )

        for offset, multiplier in enumerate(DRUM_BRIGHTNESS_PROFILE):
            leds[18 + DRUM_LEDS_PER_TYPE + offset] = (
                DRUM_COLOR,
                self.hihat_level * multiplier,
            )

        # Other: LEDs 36-53
        #
        # The new format provides:
        #   note_detected : bool
        #   brightness    : 0-255
        #   position      : 0-13
        #
        # Position 0-13 maps to 14 centre LEDs inside this 18-LED region.
        # A note also illuminates the LED immediately to each side at a
        # configurable fraction of the centre brightness.
        other = frame.get("other", {})

        # Fade every LED from previous note events.
        for other_index in range(18):
            self.other_levels[other_index] *= OTHER_FADE_PER_FRAME

            if self.other_levels[other_index] < 0.001:
                self.other_levels[other_index] = 0.0

        if other.get("note_detected", False):
            brightness = (
                clamp(float(other.get("brightness", 0)), 0, 255)
                / 255.0
            )

            position = int(
                clamp(
                    float(other.get("position", 0)),
                    0,
                    OTHER_POSITION_COUNT - 1,
                )
            )

            centre_index = OTHER_POSITION_START_OFFSET + position

            if 0 <= centre_index < 18:
                self.other_levels[centre_index] = max(
                    self.other_levels[centre_index],
                    brightness,
                )

                adjacent_level = (
                    brightness * OTHER_ADJACENT_BRIGHTNESS
                )

                for adjacent_index in (
                    centre_index - 1,
                    centre_index + 1,
                ):
                    if 0 <= adjacent_index < 18:
                        self.other_levels[adjacent_index] = max(
                            self.other_levels[adjacent_index],
                            adjacent_level,
                        )

        for other_index, level in enumerate(self.other_levels):
            leds[36 + other_index] = (OTHER_COLOR, level)

        # Vocals: LEDs 54-71
        render_spectrum_region(
            leds,
            start=54,
            spectrum=get_spectrum_values(frame, "vocals"),
            base_color=VOCAL_COLOR,
        )

        # Mirror first half across the second half.
        for i in range(HALF_LED_COUNT):
            leds[LED_COUNT - 1 - i] = leds[i]

        return leds


def draw_led(
    surface: pygame.Surface,
    x: int,
    y: int,
    base_color: tuple[int, int, int],
    brightness: float,
) -> None:
    if brightness <= 0.0:
        pygame.draw.circle(surface, (22, 22, 22), (x, y), LED_RADIUS)
        return

    actual_color = scale_color(base_color, brightness)

    if GLOW_ENABLED:
        glow_radius = round(LED_RADIUS * GLOW_RADIUS_MULTIPLIER)

        glow_surface = pygame.Surface(
            (glow_radius * 2 + 2, glow_radius * 2 + 2),
            pygame.SRCALPHA,
        )

        center = (glow_radius + 1, glow_radius + 1)

        for radius_fraction, alpha_fraction in (
            (1.0, 0.08),
            (0.72, 0.12),
            (0.45, 0.18),
        ):
            radius = max(1, round(glow_radius * radius_fraction))
            alpha = round(255 * brightness * alpha_fraction)

            pygame.draw.circle(
                glow_surface,
                (*base_color, alpha),
                center,
                radius,
            )

        surface.blit(
            glow_surface,
            (x - glow_radius - 1, y - glow_radius - 1),
        )

    pygame.draw.circle(surface, actual_color, (x, y), LED_RADIUS)


def draw_strip(
    screen: pygame.Surface,
    leds: list[tuple[tuple[int, int, int], float]],
    frame_number: int,
    frame_count: int,
    current_time: float,
    paused: bool,
    font: pygame.font.Font,
) -> None:
    screen.fill(BACKGROUND)

    usable_width = WINDOW_WIDTH - 60
    spacing = usable_width / (LED_COUNT - 1)
    y = WINDOW_HEIGHT // 2

    pygame.draw.line(
        screen,
        (35, 35, 35),
        (30, y),
        (WINDOW_WIDTH - 30, y),
        2,
    )

    for index, (base_color, brightness) in enumerate(leds):
        x = round(30 + index * spacing)
        draw_led(screen, x, y, base_color, brightness)

    status = (
        f"{'PAUSED  ' if paused else ''}"
        f"Frame {frame_number + 1}/{frame_count}   "
        f"Time {current_time:7.2f} s   "
        f"{FPS} FPS"
    )

    label = font.render(status, True, (190, 190, 190))
    screen.blit(label, (30, 25))

    help_text = "SPACE pause | LEFT/RIGHT seek 1 s | R restart | ESC quit"
    help_label = font.render(help_text, True, (110, 110, 110))
    screen.blit(help_label, (30, WINDOW_HEIGHT - 45))

    pygame.display.flip()


def rebuild_fade_state(renderer: LEDRenderer, frames: list[dict], frame_index: int) -> None:
    renderer.reset()
    start = max(0, frame_index - FPS * 2)
    for i in range(start, frame_index):
        renderer.render_frame(frames[i])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a 144-LED song visualization JSON in real time."
    )

    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to led_visualisation_data.json",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start playback at this time in seconds.",
    )

    args = parser.parse_args()
    frames = load_frames(args.json_file)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("144 LED Song Visualizer")

    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    renderer = LEDRenderer()

    frame_index = int(max(0.0, args.start) * FPS)
    frame_index = min(frame_index, len(frames) - 1)
    rebuild_fade_state(renderer, frames, frame_index)

    paused = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    frame_index = 0
                    renderer.reset()

                elif event.key == pygame.K_RIGHT:
                    frame_index = min(len(frames) - 1, frame_index + FPS)
                    rebuild_fade_state(renderer, frames, frame_index)

                elif event.key == pygame.K_LEFT:
                    frame_index = max(0, frame_index - FPS)
                    rebuild_fade_state(renderer, frames, frame_index)

        frame = frames[frame_index]
        leds = renderer.render_frame(frame)

        current_time = float(frame.get("time", frame_index / FPS))

        draw_strip(
            screen,
            leds,
            frame_index,
            len(frames),
            current_time,
            paused,
            font,
        )

        if not paused:
            frame_index += 1

            if frame_index >= len(frames):
                frame_index = 0
                renderer.reset()

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
