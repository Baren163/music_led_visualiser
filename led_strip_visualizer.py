#!/usr/bin/env python3
"""
Real-time 144-LED strip visualizer for a song's LED data.

    python led_strip_visualizer.py rebel_city

reads separated/htdemucs/<song>/led_visualisation_data.json, written by
generate_song_vis.py. That file holds everything the ESP32 gets from the
two binaries: the per-frame bass, drums and vocals, and the list of "other"
notes (manual or automatic, whichever the generator used). This preview
follows the intended firmware behaviour.

The strip is not mirrored; the layout from LED 0 is:

    0-17     Bass (fills upward with volume)
    18-26    Kick
    27-35    Hi-hat
    36       Gap
    37-84    Other: 48 LEDs, one per note position 0-47
    85       Gap
    86-143   Vocals (fills upward with volume, in the vocal colour)

Other LEDs each keep their own colour, brightness and fade. A note sets its
LED to full brightness in the note's colour and stores the note's fade;
every following frame that LED's brightness is multiplied by
(255 - fade) / 255. Fade 0 holds the LED until another note replaces it.

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

SONG_PATH = Path("separated/htdemucs")
DATA_FILENAME = "led_visualisation_data.json"

FPS = 25
LED_COUNT = 144

WINDOW_WIDTH = 1450
WINDOW_HEIGHT = 600
BACKGROUND = (8, 8, 8)

# --- LED colours (RGB, before component brightness is applied) ---------------
BASS_RGB = (255, 0, 0)
DRUM_RGB = (255, 65, 0)

# --- Component brightness, 0-255 (firmware master multipliers) ---------------
BASS_BRIGHTNESS = 255
DRUM_BRIGHTNESS = 255
OTHER_BRIGHTNESS = 255
VOCAL_BRIGHTNESS = 255

# --- LED regions --------------------------------------------------------------
BASS_START = 0
BASS_END = 17

KICK_START = 18
HIHAT_START = 27
DRUM_LEDS_PER_TYPE = 9

# One unused LED on each side of the Other region.
OTHER_LED_COUNT = 48
OTHER_START = HIHAT_START + DRUM_LEDS_PER_TYPE + 1
OTHER_END = OTHER_START + OTHER_LED_COUNT - 1

VOCAL_START = OTHER_END + 2
VOCAL_END = LED_COUNT - 1

# --- Vocals ------------------------------------------------------------------
# Volume window that maps onto the vocal LED area:
#   volume <= LOWER  -> 0 LEDs
#   volume >= UPPER  -> all vocal LEDs
VOCALS_VOLUME_LOWER_THRESHOLD = 153
VOCALS_VOLUME_UPPER_THRESHOLD = 230

# --- Drums -------------------------------------------------------------------
# Amount removed from brightness (0-255 scale) every frame.
KICK_FADE_AMOUNT = 35
HIHAT_FADE_AMOUNT = 50

# Spatial brightness profile across each 9-LED drum block (0-255).
DRUM_LED_PROFILE = [0, 64, 140, 255, 255, 255, 140, 64, 0]

# --- Other -------------------------------------------------------------------
# Other LED brightness is kept as 0-255 * 256 (16 bits) so slow fades stay
# smooth, as the firmware should.
OTHER_FULL_LEVEL = 255 << 8

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
# raw 0-255 values the firmware sends to the strip. Other and vocal colours can
# reach 255 on any channel, scaled by their component brightness.
DISPLAY_GAIN = 255 / max(
    max(_scale_rgb(BASS_RGB, BASS_BRIGHTNESS)),
    max(_scale_rgb(DRUM_RGB, DRUM_BRIGHTNESS)),
    OTHER_BRIGHTNESS,
    VOCAL_BRIGHTNESS,
    1,
)

# =============================================================================
# END CONFIGURATION
# =============================================================================


RGB = tuple[int, int, int]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_song_data(path: Path) -> tuple[list[dict], dict[int, list[dict]]]:
    """Frames, and the other notes grouped by the frame they start on."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames") if isinstance(data, dict) else None

    if not frames:
        raise SystemExit(f"Error: {path} contains no frames.")

    if "other_notes" not in data:
        raise SystemExit(
            f"Error: {path} has no 'other_notes' list. It was made by an older "
            f"generate_song_vis.py; run it again for this song."
        )

    notes_by_frame: dict[int, list[dict]] = {}

    for note in data["other_notes"]:
        notes_by_frame.setdefault(int(note["frame"]), []).append(note)

    return frames, notes_by_frame


def fade_value(value: int, fade_amount: int) -> int:
    if value <= fade_amount:
        return 0
    return value - fade_amount


def decode_rgb555(bits: int) -> RGB:
    """Unpack RGB555 (red bits 0-4, green 5-9, blue 10-14) to 0-255 channels."""
    red = bits & 0b11111
    green = (bits >> 5) & 0b11111
    blue = (bits >> 10) & 0b11111

    return tuple((channel * 255 + 15) // 31 for channel in (red, green, blue))


class LEDRenderer:
    def __init__(self, notes_by_frame: dict[int, list[dict]]) -> None:
        self.notes_by_frame = notes_by_frame
        self.reset()

    def reset(self) -> None:
        self.kick_brightness = 0
        self.hihat_brightness = 0

        # Per-LED state for the Other section.
        self.other_level = [0] * OTHER_LED_COUNT
        self.other_rgb: list[RGB] = [(0, 0, 0)] * OTHER_LED_COUNT
        self.other_fade = [0] * OTHER_LED_COUNT

    def restore_other_state(self, frame_index: int) -> None:
        """
        Rebuild the Other LEDs as they would be at frame_index, as the
        firmware does after a seek: find the last note on each LED before
        this frame and fade it by the number of frames since.
        """
        last_note: dict[int, dict] = {}

        for frame in sorted(f for f in self.notes_by_frame if f < frame_index):
            for note in self.notes_by_frame[frame]:
                last_note[note["position"]] = note

        self.other_level = [0] * OTHER_LED_COUNT

        for position, note in last_note.items():
            # Fades are applied on the frames after the note, up to and
            # including frame_index - 1 (frame_index's own fade comes when
            # it is rendered).
            age = frame_index - 1 - note["frame"]
            factor = ((255 - note["fade"]) / 255) ** age

            self.other_level[position] = int(OTHER_FULL_LEVEL * factor)
            self.other_rgb[position] = (note["r"], note["g"], note["b"])
            self.other_fade[position] = note["fade"]

    # ------------------------------------------------------------------
    # Framebuffer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set(leds: list[RGB], index: int, rgb: RGB) -> None:
        if 0 <= index < LED_COUNT:
            leds[index] = rgb

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _render_bass(self, leds: list[RGB], bass: int) -> None:
        region_length = BASS_END - BASS_START + 1
        leds_on = (bass * region_length) // 255

        colour = _scale_rgb(BASS_RGB, BASS_BRIGHTNESS)
        for i in range(leds_on):
            self._set(leds, BASS_START + i, colour)

    def _render_drum_block(self, leds: list[RGB], region_start: int, brightness: int) -> None:
        for offset in range(DRUM_LEDS_PER_TYPE):
            spatial_level = (
                brightness * DRUM_LED_PROFILE[offset] * DRUM_BRIGHTNESS
            ) // (255 * 255)

            self._set(
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

    def _render_other(self, leds: list[RGB], frame_index: int) -> None:
        # Fade every LED by its own fade value first...
        for position in range(OTHER_LED_COUNT):
            level = self.other_level[position]

            if level:
                self.other_level[position] = (level * (255 - self.other_fade[position])) // 255

        # ...then start this frame's notes at full brightness.
        for note in self.notes_by_frame.get(frame_index, ()):
            position = note["position"]
            self.other_level[position] = OTHER_FULL_LEVEL
            self.other_rgb[position] = (note["r"], note["g"], note["b"])
            self.other_fade[position] = note["fade"]

        for position in range(OTHER_LED_COUNT):
            level = ((self.other_level[position] >> 8) * OTHER_BRIGHTNESS) // 255
            self._set(
                leds,
                OTHER_START + position,
                _scale_rgb(self.other_rgb[position], level),
            )

    @staticmethod
    def _render_vocals(leds: list[RGB], vocals: dict) -> None:
        # The whole vocal LED area maps onto the volume window
        # [LOWER, UPPER], filling from VOCAL_START towards the strip end.
        vocals_length = VOCAL_END - VOCAL_START + 1
        volume = int(clamp(vocals["volume"], 0, 255))

        leds_on = int(
            (volume - VOCALS_VOLUME_LOWER_THRESHOLD)
            / (VOCALS_VOLUME_UPPER_THRESHOLD - VOCALS_VOLUME_LOWER_THRESHOLD)
            * vocals_length
        )
        leds_on = max(0, min(vocals_length, leds_on))

        rgb = decode_rgb555(int(vocals.get("color", {}).get("bits", 0)))
        colour = _scale_rgb(rgb, VOCAL_BRIGHTNESS)

        for i in range(leds_on):
            LEDRenderer._set(leds, VOCAL_START + i, colour)

    # ------------------------------------------------------------------
    # Full frame
    # ------------------------------------------------------------------

    def render_frame(self, frame: dict, frame_index: int) -> list[RGB]:
        """Returns the 144 RGB values (0-255) the firmware would send to the strip."""
        leds: list[RGB] = [(0, 0, 0)] * LED_COUNT

        bass = int(clamp(frame["bass"]["byte"], 0, 255))

        self._render_bass(leds, bass)
        self._render_drums(leds, frame["drums"])
        self._render_other(leds, frame_index)
        self._render_vocals(leds, frame["vocals"])

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

    # Blend up from the unlit grey so dim LEDs never look darker than off.
    gamma_peak = peak ** BRIGHTNESS_GAMMA
    actual_color = tuple(
        round(22 + (channel - 22) * gamma_peak) for channel in hue
    )
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
    """Restore fades after a seek: drums by replaying 2 s, Other per LED."""
    renderer.reset()
    start = max(0, frame_index - FPS * 2)

    for i in range(start, frame_index):
        renderer.render_frame(frames[i], i)

    renderer.restore_other_state(frame_index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a song's 144-LED data in real time."
    )

    parser.add_argument(
        "song",
        help="Song name, e.g. rebel_city (reads separated/htdemucs/<song>/).",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start playback at this time in seconds.",
    )

    args = parser.parse_args()

    data_path = SONG_PATH / args.song / DATA_FILENAME

    if not data_path.exists():
        raise SystemExit(
            f"Error: {data_path} not found. "
            f"Run generate_song_vis.py {args.song} first."
        )

    frames, notes_by_frame = load_song_data(data_path)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"144 LED Song Visualizer - {args.song}")

    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    renderer = LEDRenderer(notes_by_frame)

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
            leds = renderer.render_frame(frames[frame_index], frame_index)
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
