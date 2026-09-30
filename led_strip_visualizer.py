#!/usr/bin/env python3
"""
Real-time 144-LED strip visualizer for led_visualisation_data.json.

This mirrors the rendering logic of the ESP32 firmware (blink_example_main.c)
as closely as possible, using the same integer maths, region layout, colours,
fades and per-component brightness scaling.

Usage:
    pip install pygame
    python led_strip_visualizer.py path/to/led_visualisation_data.json

Controls:
    SPACE       Pause / resume
    LEFT        Seek backward 1 second
    RIGHT       Seek forward 1 second
    R           Restart
    ESC         Quit

JSON frame fields used (everything else in the file is ignored, including the
vocals spectrum and onset, which the firmware no longer uses):
    "time"   : seconds
    "bass"   : {"byte": 0-255}
    "drums"  : {"kick": bool, "kick_intensity": 0-7,
                "hihat": bool, "hihat_intensity": 0-7}
    "other"  : {"note_detected": bool, "brightness": 0-255, "position": 0-13,
                "color": 0-7 (optional, defaults to 0)}
    "vocals" : {"volume": 0-255}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pygame


# =============================================================================
# CONFIGURATION  (values below match blink_example_main.c)
# =============================================================================

FPS = 25
LED_COUNT = 144

WINDOW_WIDTH = 1450
WINDOW_HEIGHT = 600
BACKGROUND = (8, 8, 8)

# --- LED colours (RGB, before component brightness is applied) ---------------
BASS_RGB = (255, 0, 0)
DRUM_RGB = (255, 65, 0)
VOCAL_RGB = (180, 220, 0)

# One colour per Other sound type (metadata bits 5-7). Colour 0 is the most
# common sound type, so it keeps the original Other colour.
OTHER_PALETTE = [
    (200, 110, 0),     # 0 amber
    (0, 170, 255),     # 1 sky blue
    (220, 0, 180),     # 2 magenta
    (0, 220, 120),     # 3 mint
    (140, 60, 255),    # 4 violet
    (255, 40, 60),     # 5 rose
    (255, 220, 120),   # 6 warm white
    (60, 90, 255),     # 7 deep blue
]

# --- Component brightness, 0-255 (firmware master multipliers) ---------------
BASS_BRIGHTNESS = 255
DRUM_BRIGHTNESS = 255
OTHER_BRIGHTNESS = 255
VOCAL_BRIGHTNESS = 255

# --- LED regions (first half; the second half is mirrored) -------------------
BASS_START = 0
BASS_END = 17

KICK_START = 18
HIHAT_START = 27
DRUM_LEDS_PER_TYPE = 9

OTHER_MAIN_START = 38
OTHER_MAIN_END = 57
# The Other glow may use one extra LED on each side of the main region.
OTHER_RENDER_START = OTHER_MAIN_START - 1
OTHER_RENDER_END = OTHER_MAIN_END + 1
OTHER_RENDER_LENGTH = OTHER_RENDER_END - OTHER_RENDER_START + 1
OTHER_MAIN_LENGTH = OTHER_MAIN_END - OTHER_MAIN_START + 1
OTHER_POSITION_COUNT = 14

VOCAL_START = 60
VOCAL_END = 71

# --- Vocals ------------------------------------------------------------------
# Volume window that maps onto the vocal LED area:
#   volume <= LOWER  -> 0 LEDs per side
#   volume >= UPPER  -> all vocal LEDs per side
VOCALS_VOLUME_LOWER_THRESHOLD = 153
VOCALS_VOLUME_UPPER_THRESHOLD = 230

# The vocal LEDs fill outwards from the centre of the strip: LED 71 (and its
# mirror, LED 72) light first, then 70/73, and so on down to LED 60/83.

# --- Drums -------------------------------------------------------------------
# Amount removed from brightness (0-255 scale) every frame.
KICK_FADE_AMOUNT = 35
HIHAT_FADE_AMOUNT = 50

# Spatial brightness profile across each 9-LED drum block (0-255).
DRUM_LED_PROFILE = [0, 64, 140, 255, 255, 255, 140, 64, 0]

# --- Other -------------------------------------------------------------------
OTHER_FADE_AMOUNT = 45          # removed from every Other LED per frame
OTHER_ADJACENT_SCALE = 64       # neighbours get 64/255 of the note brightness

# --- On-screen appearance (does not exist in the firmware) -------------------
LED_RADIUS = 5
GLOW_ENABLED = True
GLOW_RADIUS_MULTIPLIER = 3.2
BRIGHTNESS_GAMMA = 0.85


def _scale_rgb(rgb: tuple[int, int, int], brightness: int) -> tuple[int, int, int]:
    return tuple((channel * brightness) // 255 for channel in rgb)


# The real strip runs at low component brightness (e.g. vocals at 30/255), which
# would look very dark on a monitor. DISPLAY_GAIN scales every LED by the same
# factor so the brightest possible LED reaches full screen brightness while all
# the relative brightness between components is preserved. Set to 1.0 to see the
# raw 0-255 values the firmware sends to the strip.
DISPLAY_GAIN = 255 / max(
    max(_scale_rgb(BASS_RGB, BASS_BRIGHTNESS)),
    max(_scale_rgb(DRUM_RGB, DRUM_BRIGHTNESS)),
    max(max(_scale_rgb(rgb, OTHER_BRIGHTNESS)) for rgb in OTHER_PALETTE),
    max(_scale_rgb(VOCAL_RGB, VOCAL_BRIGHTNESS)),
)

# =============================================================================
# END CONFIGURATION
# =============================================================================


RGB = tuple[int, int, int]


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


def fade_value(value: int, fade_amount: int) -> int:
    if value <= fade_amount:
        return 0
    return value - fade_amount


def other_position_to_led(position: int) -> int:
    """Index into the Other render region (matches other_position_to_led in C)."""
    position = min(position, OTHER_POSITION_COUNT - 1)
    main_offset = (position * (OTHER_MAIN_LENGTH - 1) + 6) // 13
    return main_offset + 1


class LEDRenderer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.kick_brightness = 0
        self.hihat_brightness = 0
        self.other_led_brightness = [0] * OTHER_RENDER_LENGTH
        self.other_led_color = [0] * OTHER_RENDER_LENGTH

    # ------------------------------------------------------------------
    # Framebuffer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_mirrored(leds: list[RGB], first_half_index: int, rgb: RGB) -> None:
        if 0 <= first_half_index < LED_COUNT:
            leds[first_half_index] = rgb

        mirrored_index = (LED_COUNT - 1) - first_half_index
        if 0 <= mirrored_index < LED_COUNT:
            leds[mirrored_index] = rgb

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _render_bass(self, leds: list[RGB], bass: int) -> None:
        region_length = BASS_END - BASS_START + 1
        leds_on = (bass * region_length) // 255

        colour = _scale_rgb(BASS_RGB, BASS_BRIGHTNESS)
        for i in range(leds_on):
            self._set_mirrored(leds, BASS_START + i, colour)

    def _render_drum_block(self, leds: list[RGB], region_start: int, brightness: int) -> None:
        for offset in range(DRUM_LEDS_PER_TYPE):
            spatial_level = (
                brightness * DRUM_LED_PROFILE[offset] * DRUM_BRIGHTNESS
            ) // (255 * 255)

            self._set_mirrored(
                leds,
                region_start + offset,
                _scale_rgb(DRUM_RGB, spatial_level),
            )

    def _render_drums(self, leds: list[RGB], drums: dict) -> None:
        # Fade previous beats first, then raise to any new beat.
        self.kick_brightness = fade_value(self.kick_brightness, KICK_FADE_AMOUNT)
        self.hihat_brightness = fade_value(self.hihat_brightness, HIHAT_FADE_AMOUNT)

        if drums["kick"]:
            intensity = int(clamp(drums["kick_intensity"], 0, 7))
            new_brightness = (intensity * 255) // 7
            if new_brightness > self.kick_brightness:
                self.kick_brightness = new_brightness

        if drums["hihat"]:
            intensity = int(clamp(drums["hihat_intensity"], 0, 7))
            new_brightness = (intensity * 255) // 7
            if new_brightness > self.hihat_brightness:
                self.hihat_brightness = new_brightness

        self._render_drum_block(leds, KICK_START, self.kick_brightness)
        self._render_drum_block(leds, HIHAT_START, self.hihat_brightness)

    def _raise_other_led(self, offset: int, brightness: int, color: int) -> None:
        if offset < 0 or offset >= OTHER_RENDER_LENGTH:
            return
        if brightness > self.other_led_brightness[offset]:
            self.other_led_brightness[offset] = brightness
            self.other_led_color[offset] = color

    def _render_other(self, leds: list[RGB], other: dict) -> None:
        # Fade all previous Other LEDs first.
        for i in range(OTHER_RENDER_LENGTH):
            self.other_led_brightness[i] = fade_value(
                self.other_led_brightness[i], OTHER_FADE_AMOUNT
            )

        note_detected = other["note_detected"]
        brightness = int(clamp(other["brightness"], 0, 255))
        position = int(other["position"])
        color = int(clamp(other.get("color", 0), 0, len(OTHER_PALETTE) - 1))

        if note_detected and 0 <= position < OTHER_POSITION_COUNT and brightness > 0:
            main_offset = other_position_to_led(position)
            adjacent_brightness = (brightness * OTHER_ADJACENT_SCALE) // 255

            self._raise_other_led(main_offset, brightness, color)
            self._raise_other_led(main_offset - 1, adjacent_brightness, color)
            self._raise_other_led(main_offset + 1, adjacent_brightness, color)

        for offset in range(OTHER_RENDER_LENGTH):
            level = (self.other_led_brightness[offset] * OTHER_BRIGHTNESS) // 255
            self._set_mirrored(
                leds,
                OTHER_RENDER_START + offset,
                _scale_rgb(OTHER_PALETTE[self.other_led_color[offset]], level),
            )

    @staticmethod
    def _render_vocals(leds: list[RGB], volume: int) -> None:
        # The whole vocal LED area maps onto the volume window
        # [LOWER, UPPER], and is mirrored across the middle of the strip.
        vocals_length = VOCAL_END - VOCAL_START + 1

        leds_on = int(
            (volume - VOCALS_VOLUME_LOWER_THRESHOLD)
            / (VOCALS_VOLUME_UPPER_THRESHOLD - VOCALS_VOLUME_LOWER_THRESHOLD)
            * vocals_length
        )
        leds_on = max(0, min(vocals_length, leds_on))

        colour = _scale_rgb(VOCAL_RGB, VOCAL_BRIGHTNESS)
        for i in range(leds_on):
            LEDRenderer._set_mirrored(leds, VOCAL_END - i, colour)

    # ------------------------------------------------------------------
    # Full frame
    # ------------------------------------------------------------------

    def render_frame(self, frame: dict) -> list[RGB]:
        """Returns the 144 RGB values (0-255) the firmware would send to the strip."""
        leds: list[RGB] = [(0, 0, 0)] * LED_COUNT

        bass = int(clamp(frame["bass"]["byte"], 0, 255))
        volume = int(clamp(frame["vocals"]["volume"], 0, 255))

        self._render_bass(leds, bass)
        self._render_drums(leds, frame["drums"])
        self._render_other(leds, frame["other"])
        self._render_vocals(leds, volume)

        return leds


def draw_led(surface: pygame.Surface, x: int, y: int, rgb: RGB) -> None:
    # Apply the on-screen gain while keeping the colour's hue.
    normalized = [channel / 255.0 * DISPLAY_GAIN for channel in rgb]
    peak = max(normalized)

    if peak <= 0.0:
        pygame.draw.circle(surface, (22, 22, 22), (x, y), LED_RADIUS)
        return

    if peak > 1.0:
        normalized = [channel / peak for channel in normalized]
        peak = 1.0

    # Full-brightness version of this LED's colour (max channel = 255).
    hue = [channel / peak * 255.0 for channel in normalized]

    gamma_peak = peak ** BRIGHTNESS_GAMMA
    actual_color = tuple(round(channel * gamma_peak) for channel in hue)
    base_color = tuple(round(channel) for channel in hue)

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
            alpha = round(255 * peak * alpha_fraction)

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
    leds: list[RGB],
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

    for index, rgb in enumerate(leds):
        x = round(30 + index * spacing)
        draw_led(screen, x, y, rgb)

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

    # Rendering advances the fade state, so a frame is only rendered once:
    # when playing, or once after a seek/restart while paused.
    needs_render = True
    displayed_index = frame_index
    leds: list[RGB] = [(0, 0, 0)] * LED_COUNT

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
                    needs_render = True

                elif event.key == pygame.K_RIGHT:
                    frame_index = min(len(frames) - 1, frame_index + FPS)
                    rebuild_fade_state(renderer, frames, frame_index)
                    needs_render = True

                elif event.key == pygame.K_LEFT:
                    frame_index = max(0, frame_index - FPS)
                    rebuild_fade_state(renderer, frames, frame_index)
                    needs_render = True

        if needs_render or not paused:
            leds = renderer.render_frame(frames[frame_index])
            displayed_index = frame_index
            needs_render = False

            if not paused:
                frame_index += 1

                if frame_index >= len(frames):
                    frame_index = 0
                    renderer.reset()

        frame = frames[displayed_index]
        current_time = frame["time"]

        draw_strip(
            screen,
            leds,
            displayed_index,
            len(frames),
            current_time,
            paused,
            font,
        )

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
