#!/usr/bin/env python3
"""
manual_other_recorder.py

Play a song and record "other" LED notes by hand.

Open an MP3, play it, and press the number keys to place notes on the 48
"other" LEDs at the current song time. Notes are saved immediately to
other_manual.json next to the MP3 (normally in the song's separated stem
folder, so generate_song_vis.py can find it) and shown on the LED preview
straight away. Existing notes are loaded when the MP3 is opened.

Keys:
    1-9, 0      Place a note on the 10 LEDs under the key window
                (1 = leftmost, 0 = rightmost)
    Space       Play / pause
    Left/Right  Seek -5 s / +5 s
    R           Restart
    Ctrl+Z      Undo last note

Notes file (other_manual.json):

    {
      "format": "other_manual_v1",
      "notes": [
        {"frame": 1234, "time": 49.36, "position": 12,
         "r": 255, "g": 120, "b": 0, "fade": 40},
        ...
      ]
    }

    frame     25 Hz frame index the note starts on
    position  LED 0-47 within the other section
    r, g, b   colour 0-255
    fade      0-255. Each frame after the note, the LED's brightness is
              multiplied by (255 - fade) / 255. 0 holds the LED until
              another note replaces it; 255 lights it for one frame only.

A later note on the same frame and LED replaces the earlier one.

Usage:

    python manual_other_recorder.py
"""

import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import librosa
import numpy as np
import pygame


# =============================================================================
# CONFIGURATION
# =============================================================================

FRAME_RATE = 25.0

LED_COUNT = 48

# Keyboard order of the number keys, left to right.
NOTE_KEYS = "1234567890"

SEEK_SECONDS = 5.0
SPEEDS = (1.0, 0.75, 0.5, 0.25)

NOTES_FILENAME = "other_manual.json"
NOTES_FORMAT = "other_manual_v1"

# A notes file whose duration differs from the opened MP3 by more than this
# belongs to another song and is not loaded or overwritten.
DURATION_TOLERANCE_SECONDS = 0.5

DEFAULT_COLOR = (255, 120, 0)
DEFAULT_FADE = 40

# Smaller = lower audio latency, but may crackle on slow machines.
AUDIO_BUFFER_SAMPLES = 512

UPDATE_INTERVAL_MS = 20

# --- Appearance ---------------------------------------------------------------
BACKGROUND = "#141418"
PANEL = "#1d1d23"
FOREGROUND = "#e8e8ee"
MUTED = "#8a8a96"
ACCENT = "#ff69c8"
LED_OFF_RGB = (42, 42, 50)
LED_OFF = "#2a2a32"
TRACK = "#34343e"
FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_LARGE = ("Segoe UI", 14, "bold")

# =============================================================================
# END CONFIGURATION
# =============================================================================


def format_time(seconds):
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"


def rgb_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


def fade_description(fade):
    """How long a note takes to fade to 5% brightness."""
    if fade <= 0:
        return "holds until replaced"

    if fade >= 255:
        return "1 frame"

    frames = math.log(0.05) / math.log((255 - fade) / 255)
    return f"about {frames / FRAME_RATE:.2f} s"


# -----------------------------------------------------------------------------
# Notes file
# -----------------------------------------------------------------------------


class NotesFile:
    def __init__(self, path, source_name, duration):
        self.path = path
        self.source_name = source_name
        self.duration = duration
        self.notes = []
        self.undo_stack = []
        self.by_position = {}

    def load(self):
        """Load existing notes. Raises ValueError if the file doesn't fit."""
        if not self.path.exists():
            self._reindex()
            return

        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("format") != NOTES_FORMAT:
            raise ValueError(
                f"{self.path.name} has format {data.get('format')!r}, "
                f"expected {NOTES_FORMAT!r}."
            )

        file_duration = float(data.get("duration", self.duration))

        if abs(file_duration - self.duration) > DURATION_TOLERANCE_SECONDS:
            raise ValueError(
                f"{self.path.name} was recorded for a {file_duration:.2f} s "
                f"song ({data.get('source')}), but this MP3 is "
                f"{self.duration:.2f} s long. It probably belongs to another "
                f"song, so it was not loaded or changed."
            )

        self.notes = [
            {
                "frame": int(note["frame"]),
                "time": round(int(note["frame"]) / FRAME_RATE, 4),
                "position": int(note["position"]),
                "r": int(note["r"]),
                "g": int(note["g"]),
                "b": int(note["b"]),
                "fade": int(note["fade"]),
            }
            for note in data.get("notes", [])
        ]

        self._sort()
        self._reindex()

    def add(self, frame, position, rgb, fade):
        note = {
            "frame": frame,
            "time": round(frame / FRAME_RATE, 4),
            "position": position,
            "r": rgb[0],
            "g": rgb[1],
            "b": rgb[2],
            "fade": fade,
        }

        replaced = None

        for i, existing in enumerate(self.notes):
            if existing["frame"] == frame and existing["position"] == position:
                replaced = self.notes.pop(i)
                break

        self.notes.append(note)
        self.undo_stack.append((note, replaced))
        self._changed()

    def undo(self):
        if not self.undo_stack:
            return False

        added, replaced = self.undo_stack.pop()
        self.notes.remove(added)

        if replaced is not None:
            self.notes.append(replaced)

        self._changed()
        return True

    def brightness_and_color(self, position, frame):
        """Colour and 0-1 brightness of an LED at a frame."""
        entries = self.by_position.get(position)

        if not entries:
            return None, 0.0

        # Last note on this LED at or before the frame.
        index = np.searchsorted(entries[0], frame, side="right") - 1

        if index < 0:
            return None, 0.0

        note = entries[1][index]
        age = frame - note["frame"]
        brightness = ((255 - note["fade"]) / 255) ** age

        return (note["r"], note["g"], note["b"]), brightness

    def _sort(self):
        self.notes.sort(key=lambda note: (note["frame"], note["position"]))

    def _reindex(self):
        self.by_position = {}

        for note in self.notes:
            self.by_position.setdefault(note["position"], []).append(note)

        for position, entries in self.by_position.items():
            self.by_position[position] = (
                np.array([note["frame"] for note in entries]),
                entries,
            )

    def _changed(self):
        self._sort()
        self._reindex()
        self.save()

    def save(self):
        data = {
            "format": NOTES_FORMAT,
            "source": self.source_name,
            "duration": round(self.duration, 5),
            "frame_rate": FRAME_RATE,
            "led_count": LED_COUNT,
            "fade_definition": (
                "each frame after a note, brightness *= (255 - fade) / 255; "
                "0 holds until replaced"
            ),
            "notes": self.notes,
        }

        # Write to a temporary file first so a crash can't corrupt the notes.
        temporary = self.path.with_suffix(".json.tmp")

        with temporary.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        temporary.replace(self.path)


# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------


def to_int16_stereo(audio):
    """(channels, samples) float -> contiguous (samples, 2) int16."""
    if audio.shape[0] == 1:
        audio = np.vstack([audio, audio])

    audio = audio[:2]

    return np.ascontiguousarray(
        (np.clip(audio.T, -1.0, 1.0) * 32767).astype(np.int16)
    )


class AudioPlayer:
    """
    Plays a decoded song with pygame's mixer. Slower speeds use
    time-stretched copies (pitch preserved) prepared in the background.
    """

    def __init__(self):
        self.sample_rate = None
        self.audio = None
        self.duration = 0.0
        self.versions = {}

        self.speed = 1.0
        self.playing = False
        self.anchor_song_time = 0.0
        self.anchor_wall_time = 0.0
        self.sound = None
        self.channel = None

    def load(self, path):
        self.stop()

        audio, sample_rate = librosa.load(path, sr=None, mono=False)

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        if pygame.mixer.get_init() and pygame.mixer.get_init()[0] != sample_rate:
            pygame.mixer.quit()

        if not pygame.mixer.get_init():
            pygame.mixer.init(
                frequency=sample_rate,
                size=-16,
                channels=2,
                buffer=AUDIO_BUFFER_SAMPLES,
            )

        self.audio = audio
        self.sample_rate = sample_rate
        self.duration = audio.shape[1] / sample_rate
        self.versions = {1.0: to_int16_stereo(audio)}
        self.speed = 1.0
        self.anchor_song_time = 0.0

    def make_version(self, speed):
        """Slow, run in a background thread."""
        stretched = librosa.effects.time_stretch(self.audio, rate=speed)
        return to_int16_stereo(stretched)

    def position(self):
        if not self.playing:
            return self.anchor_song_time

        elapsed = time.perf_counter() - self.anchor_wall_time
        return min(self.duration, self.anchor_song_time + elapsed * self.speed)

    def play(self):
        if self.audio is None or self.anchor_song_time >= self.duration:
            return

        samples = self.versions[self.speed]
        start = int(self.anchor_song_time / self.speed * self.sample_rate)

        self.sound = pygame.sndarray.make_sound(samples[start:])
        self.channel = self.sound.play()
        self.anchor_wall_time = time.perf_counter()
        self.playing = True

    def pause(self):
        self.anchor_song_time = self.position()
        self.stop()

    def stop(self):
        if self.channel is not None:
            self.channel.stop()

        self.channel = None
        self.sound = None
        self.playing = False

    def seek(self, song_time):
        was_playing = self.playing
        self.stop()
        self.anchor_song_time = min(max(0.0, song_time), self.duration)

        if was_playing:
            self.play()

    def set_speed(self, speed):
        was_playing = self.playing
        self.pause()
        self.speed = speed

        if was_playing:
            self.play()

    def check_end(self):
        if self.playing and self.position() >= self.duration:
            self.stop()
            self.anchor_song_time = self.duration


# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------


class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.player = AudioPlayer()
        self.notes = None

        self.color = DEFAULT_COLOR
        self.held_keys = set()

        self.song_token = 0
        self.pending_speed = None
        self.preparing = set()
        self.results = queue.Queue()

        self.fade_var = tk.IntVar(value=DEFAULT_FADE)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.window_var = tk.IntVar(value=0)

        root.title("Manual Other Recorder")
        root.configure(bg=BACKGROUND)
        root.geometry("1180x500")
        root.minsize(980, 480)

        self._build_header()
        self._build_body()
        self._bind_keys()

        self._draw_led_strip()
        self._update_fade_label()
        self._tick()

    # --- Layout -------------------------------------------------------------

    def _label(self, parent, text="", font=FONT, fg=FOREGROUND, **kwargs):
        return tk.Label(parent, text=text, font=font, fg=fg, bg=parent["bg"], **kwargs)

    def _button(self, parent, text, command, width=None):
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            font=FONT_BOLD,
            fg=FOREGROUND,
            bg="#2c2c36",
            activebackground="#3a3a46",
            activeforeground=FOREGROUND,
            relief="flat",
            padx=10,
            pady=4,
            takefocus=0,
        )

    def _scale(self, parent, variable, from_, to, command, length):
        return tk.Scale(
            parent,
            variable=variable,
            from_=from_,
            to=to,
            orient="horizontal",
            command=command,
            length=length,
            showvalue=False,
            bg=parent["bg"],
            fg=FOREGROUND,
            troughcolor=TRACK,
            highlightthickness=0,
            sliderrelief="flat",
            activebackground=ACCENT,
            takefocus=0,
        )

    def _build_header(self):
        header = tk.Frame(self.root, bg=BACKGROUND)
        header.pack(fill="x", padx=16, pady=(14, 6))

        self._button(header, "Open MP3...", self.open_song).pack(side="left")

        info = tk.Frame(header, bg=BACKGROUND)
        info.pack(side="left", padx=14)

        self.song_label = self._label(info, "No song loaded", font=FONT_LARGE)
        self.song_label.pack(anchor="w")

        self.file_label = self._label(info, "", fg=MUTED)
        self.file_label.pack(anchor="w")

        self.status_label = self._label(header, "", fg=ACCENT)
        self.status_label.pack(side="right")

    def _build_body(self):
        body = tk.Frame(self.root, bg=BACKGROUND)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self._build_sidebar(body)

        main = tk.Frame(body, bg=PANEL)
        main.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self._build_led_strip(main)
        self._build_transport(main)

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=PANEL, width=250)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        pad = {"padx": 14, "anchor": "w"}

        self._label(sidebar, "NOTE COLOUR", font=FONT_BOLD, fg=MUTED).pack(pady=(14, 4), **pad)

        color_row = tk.Frame(sidebar, bg=PANEL)
        color_row.pack(fill="x", **pad)

        self.swatch = tk.Canvas(
            color_row, width=56, height=30, bg=PANEL, highlightthickness=0
        )
        self.swatch.pack(side="left")
        self.swatch_rect = self.swatch.create_rectangle(
            2, 2, 54, 28, fill=rgb_hex(self.color), outline=""
        )

        self._button(color_row, "Choose...", self.choose_color).pack(side="left", padx=8)

        self.color_label = self._label(sidebar, "", fg=MUTED)
        self.color_label.pack(**pad)
        self._update_color_label()

        self._label(sidebar, "FADE", font=FONT_BOLD, fg=MUTED).pack(pady=(16, 0), **pad)

        self._scale(
            sidebar, self.fade_var, 0, 255,
            lambda _value: self._update_fade_label(), 220,
        ).pack(**pad)

        self.fade_label = self._label(sidebar, "", fg=MUTED)
        self.fade_label.pack(**pad)

        self._label(sidebar, "SPEED", font=FONT_BOLD, fg=MUTED).pack(pady=(16, 2), **pad)

        speed_row = tk.Frame(sidebar, bg=PANEL)
        speed_row.pack(fill="x", **pad)

        for speed in SPEEDS:
            tk.Radiobutton(
                speed_row,
                text=f"{speed:g}x",
                value=speed,
                variable=self.speed_var,
                command=self.choose_speed,
                indicatoron=False,
                font=FONT_BOLD,
                fg=FOREGROUND,
                bg="#2c2c36",
                selectcolor="#55304a",
                activebackground="#3a3a46",
                activeforeground=FOREGROUND,
                relief="flat",
                padx=8,
                pady=3,
                takefocus=0,
            ).pack(side="left", padx=(0, 4))

        self._label(sidebar, "NOTES", font=FONT_BOLD, fg=MUTED).pack(pady=(16, 2), **pad)

        self.count_label = self._label(sidebar, "0 notes")
        self.count_label.pack(**pad)

        self._button(sidebar, "Undo last note", self.undo).pack(pady=(6, 0), **pad)

        self._label(
            sidebar,
            "Keys 1-9, 0: place note\n"
            "Space: play / pause\n"
            "Left / Right: -5 s / +5 s\n"
            "R: restart   Ctrl+Z: undo",
            fg=MUTED,
            justify="left",
        ).pack(side="bottom", pady=14, **pad)

    def _build_led_strip(self, parent):
        self._label(parent, "OTHER SECTION (48 LEDs)", font=FONT_BOLD, fg=MUTED).pack(
            anchor="w", padx=16, pady=(14, 0)
        )

        self.led_canvas = tk.Canvas(parent, height=110, bg=PANEL, highlightthickness=0)
        self.led_canvas.pack(fill="x", padx=16, pady=(4, 0))
        self.led_canvas.bind("<Configure>", lambda _event: self._draw_led_strip())

        window_row = tk.Frame(parent, bg=PANEL)
        window_row.pack(fill="x", padx=16)

        self._label(window_row, "Key window", fg=MUTED).pack(side="left")

        self.window_scale = self._scale(
            window_row, self.window_var, 0, LED_COUNT - len(NOTE_KEYS),
            lambda _value: self._draw_led_strip(), 10,
        )
        self.window_scale.pack(side="left", fill="x", expand=True, padx=10)

        self.window_label = self._label(window_row, "", fg=MUTED, width=18)
        self.window_label.pack(side="left")

    def _build_transport(self, parent):
        transport = tk.Frame(parent, bg=PANEL)
        transport.pack(fill="x", padx=16, pady=(26, 0))

        self._button(transport, "Restart", self.restart).pack(side="left")
        self._button(transport, "-5 s", lambda: self.seek_relative(-SEEK_SECONDS)).pack(side="left", padx=(8, 0))

        self.play_button = self._button(transport, "Play", self.toggle_play, width=8)
        self.play_button.pack(side="left", padx=8)

        self._button(transport, "+5 s", lambda: self.seek_relative(SEEK_SECONDS)).pack(side="left")

        self.time_label = self._label(transport, "00:00.00 / 00:00.00", font=FONT_LARGE)
        self.time_label.pack(side="right")

        self.seek_canvas = tk.Canvas(parent, height=44, bg=PANEL, highlightthickness=0)
        self.seek_canvas.pack(fill="x", padx=16, pady=(10, 14))
        self.seek_canvas.bind("<Configure>", lambda _event: self._draw_seek_static())
        self.seek_canvas.bind("<Button-1>", self._seek_click)
        self.seek_canvas.bind("<B1-Motion>", self._seek_click)

        self._draw_seek_static()

    # --- Keys ---------------------------------------------------------------

    def _bind_keys(self):
        # Stop buttons, radio buttons and sliders from reacting to these
        # keys, so they always control playback.
        for widget_class in ("Button", "Radiobutton", "Scale"):
            for sequence in ("<space>", "<Left>", "<Right>", "<Up>", "<Down>"):
                self.root.unbind_class(widget_class, sequence)

        self.root.bind_all("<KeyPress>", self._key_press)
        self.root.bind_all("<KeyRelease>", self._key_release)
        self.root.bind_all("<Control-z>", lambda _event: self.undo())

    def _key_press(self, event):
        key = event.keysym

        # Holding a key sends repeated presses; only the first one counts.
        if key in self.held_keys:
            return "break"

        self.held_keys.add(key)

        if event.char and event.char in NOTE_KEYS and not event.state & 0x4:
            self.place_note(NOTE_KEYS.index(event.char))
        elif key == "space":
            self.toggle_play()
        elif key == "Left":
            self.seek_relative(-SEEK_SECONDS)
        elif key == "Right":
            self.seek_relative(SEEK_SECONDS)
        elif key in ("r", "R"):
            self.restart()

        return "break"

    def _key_release(self, event):
        self.held_keys.discard(event.keysym)

    # --- Actions ------------------------------------------------------------

    def open_song(self):
        path = filedialog.askopenfilename(
            title="Open song",
            filetypes=[("MP3 files", "*.mp3"), ("Audio files", "*.mp3 *.wav *.flac *.ogg")],
        )

        if path:
            self.load_song(Path(path))

    def load_song(self, path):
        self.status_label.config(text="Loading...")
        self.root.update_idletasks()

        try:
            self.player.load(path)
        except Exception as exc:
            self.status_label.config(text="")
            messagebox.showerror("Could not open song", str(exc))
            return

        notes = NotesFile(path.with_name(NOTES_FILENAME), path.name, self.player.duration)

        try:
            notes.load()
        except Exception as exc:
            self.player.stop()
            self.player.audio = None
            self.notes = None
            self.song_label.config(text="No song loaded")
            self.file_label.config(text="")
            self.status_label.config(text="")
            messagebox.showerror("Notes file problem", str(exc))
            return

        self.notes = notes
        self.song_token += 1
        self.preparing.clear()
        self.pending_speed = None
        self.speed_var.set(1.0)

        self.song_label.config(text=path.name)
        self.file_label.config(text=f"Notes: {notes.path}")
        self.status_label.config(text="")

        self._update_count()
        self._draw_seek_static()

    def choose_color(self):
        result = colorchooser.askcolor(color=rgb_hex(self.color), title="Note colour")

        if result and result[0]:
            self.color = tuple(int(round(c)) for c in result[0])
            self.swatch.itemconfig(self.swatch_rect, fill=rgb_hex(self.color))
            self._update_color_label()

    def choose_speed(self):
        speed = float(self.speed_var.get())

        if self.player.audio is None:
            self.speed_var.set(1.0)
            return

        if speed in self.player.versions:
            self.pending_speed = None
            self.player.set_speed(speed)
            self.status_label.config(text="")
            return

        # Prepare the stretched version in the background, then switch.
        self.pending_speed = speed
        self.status_label.config(text=f"Preparing {speed:g}x speed...")

        if speed not in self.preparing:
            self.preparing.add(speed)
            token = self.song_token

            def work():
                try:
                    samples = self.player.make_version(speed)
                    self.results.put((token, speed, samples, None))
                except Exception as exc:
                    self.results.put((token, speed, None, exc))

            threading.Thread(target=work, daemon=True).start()

    def toggle_play(self):
        if self.player.audio is None:
            return

        if self.player.playing:
            self.player.pause()
        else:
            if self.player.position() >= self.player.duration:
                self.player.seek(0.0)

            self.player.play()

    def seek_relative(self, seconds):
        if self.player.audio is not None:
            self.player.seek(self.player.position() + seconds)

    def restart(self):
        if self.player.audio is None:
            return

        self.player.seek(0.0)

        if not self.player.playing:
            self.player.play()

    def place_note(self, key_index):
        if self.notes is None:
            return

        last_frame = max(0, int(self.player.duration * FRAME_RATE) - 1)
        frame = min(last_frame, int(round(self.player.position() * FRAME_RATE)))
        position = self.window_var.get() + key_index

        self.notes.add(frame, position, self.color, int(self.fade_var.get()))
        self._update_count()
        self._draw_seek_static()

    def undo(self):
        if self.notes is not None and self.notes.undo():
            self._update_count()
            self._draw_seek_static()

        return "break"

    # --- Drawing ------------------------------------------------------------

    def _update_color_label(self):
        self.color_label.config(text="RGB {}, {}, {}".format(*self.color))

    def _update_fade_label(self):
        fade = int(self.fade_var.get())
        self.fade_label.config(text=f"{fade}  ({fade_description(fade)})")

    def _update_count(self):
        count = len(self.notes.notes) if self.notes else 0
        self.count_label.config(text=f"{count} note{'s' if count != 1 else ''}")

    def _draw_led_strip(self):
        canvas = self.led_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 200)
        spacing = (width - 20) / LED_COUNT
        radius = max(4, min(9, spacing * 0.36))
        y = 36

        start = self.window_var.get()
        end = start + len(NOTE_KEYS) - 1

        # Highlight behind the LEDs the number keys currently cover.
        left = 10 + start * spacing
        right = 10 + (end + 1) * spacing
        canvas.create_rectangle(left, 8, right, 94, fill="#262630", outline=ACCENT)

        self.led_items = []

        for index in range(LED_COUNT):
            x = 10 + (index + 0.5) * spacing
            self.led_items.append(
                canvas.create_oval(
                    x - radius, y - radius, x + radius, y + radius,
                    fill=LED_OFF, outline="",
                )
            )

            if index % 8 == 0:
                canvas.create_text(x, 104, text=str(index), fill=MUTED, font=("Segoe UI", 8))

        for key_index, key in enumerate(NOTE_KEYS):
            x = 10 + (start + key_index + 0.5) * spacing
            canvas.create_text(x, 70, text=key, fill=FOREGROUND, font=FONT_BOLD)

        self.window_label.config(text=f"keys cover LEDs {start}-{end}")

    def _draw_seek_static(self):
        canvas = self.seek_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 200)
        self.seek_left = 6
        self.seek_right = width - 6
        y = 22

        canvas.create_line(self.seek_left, y, self.seek_right, y, fill=TRACK, width=6, capstyle="round")

        duration = self.player.duration

        # One tick per note, in its colour.
        if self.notes is not None and duration > 0:
            span = self.seek_right - self.seek_left

            for note in self.notes.notes:
                x = self.seek_left + note["time"] / duration * span
                canvas.create_line(
                    x, y - 14, x, y - 5,
                    fill=rgb_hex((note["r"], note["g"], note["b"])),
                    width=2,
                )

        self.seek_progress = canvas.create_line(
            self.seek_left, y, self.seek_left, y, fill=ACCENT, width=6, capstyle="round"
        )
        self.seek_head = canvas.create_oval(0, 0, 0, 0, fill=FOREGROUND, outline="")

    def _seek_click(self, event):
        if self.player.audio is None:
            return

        span = max(1, self.seek_right - self.seek_left)
        fraction = min(max((event.x - self.seek_left) / span, 0.0), 1.0)
        self.player.seek(fraction * self.player.duration)

    def _tick(self):
        self._collect_results()
        self.player.check_end()

        position = self.player.position()
        duration = self.player.duration

        self.time_label.config(text=f"{format_time(position)} / {format_time(duration)}")
        self.play_button.config(text="Pause" if self.player.playing else "Play")

        # Seek bar.
        fraction = position / duration if duration > 0 else 0.0
        x = self.seek_left + fraction * (self.seek_right - self.seek_left)
        self.seek_canvas.coords(self.seek_progress, self.seek_left, 22, x, 22)
        self.seek_canvas.coords(self.seek_head, x - 7, 15, x + 7, 29)

        # LEDs.
        frame = int(round(position * FRAME_RATE))

        for index, item in enumerate(self.led_items):
            colour = LED_OFF

            if self.notes is not None:
                rgb, brightness = self.notes.brightness_and_color(index, frame)

                # Blend from the "off" colour so dim LEDs never look darker
                # than unlit ones.
                if rgb is not None:
                    colour = rgb_hex(tuple(
                        off + (c - off) * brightness
                        for off, c in zip(LED_OFF_RGB, rgb)
                    ))

            self.led_canvas.itemconfig(item, fill=colour)

        self.root.after(UPDATE_INTERVAL_MS, self._tick)

    def _collect_results(self):
        while True:
            try:
                token, speed, samples, error = self.results.get_nowait()
            except queue.Empty:
                return

            self.preparing.discard(speed)

            if token != self.song_token:
                continue

            if error is not None:
                self.status_label.config(text="")
                messagebox.showerror("Speed change failed", str(error))
                continue

            self.player.versions[speed] = samples

            if self.pending_speed == speed:
                self.pending_speed = None
                self.player.set_speed(speed)
                self.status_label.config(text="")


def main():
    root = tk.Tk()
    RecorderApp(root)
    root.mainloop()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
