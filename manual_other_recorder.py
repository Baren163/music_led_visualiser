#!/usr/bin/env python3
"""
manual_other_recorder.py

Play a song and record "other" LED notes by hand.

Open an MP3, play it, and press the number keys to place notes on the 48
"other" LEDs at the current song time. Notes are saved immediately to
separated/htdemucs/<song name>/other_manual.json, where the song name is the
MP3's file name without ".mp3" (rebel_city.mp3 -> rebel_city), and shown on
the LED preview straight away. Existing notes are loaded from the same place
when the MP3 is opened. Opening an other.mp3 stem asks for the song name
(e.g. worry_child) instead, since the file name doesn't say.

Keys:
    1-9, 0      Place a note on the 10 LEDs under the key window
                (1 = leftmost, 0 = rightmost)
    Space       Play / pause
    Left/Right  Seek -5 s / +5 s
    R           Restart
    Ctrl+Z      Undo last note

The note width (1-6) sets how many LEDs one key press lights: the key's
LED and the ones after it. Each LED is stored as its own note, but one undo
removes the whole key press.

With "Snap to Nearest Onset" ticked, each note moves to the frame with the
highest overall onset (from the song's other_onsets.json) within one frame
either side.

Keys pressed within SNAP_FRAMES frames of the first key in a group are
placed on that first key's frame, so chords line up. The audio offset
slider places notes that many milliseconds before the key press, to
make up for reaction time.

The "Advanced" button opens a window listing every note. Notes can be
selected by clicking, or with filters (time range, LED range, fade range,
colour is / is not) that select, add to, remove from or narrow down the
selection. The selected notes can then be shifted in time (in whole 40 ms
frames), moved to a time, shifted along the LEDs, recoloured, given a new
fade or deleted. Each edit is one undo step. The window only does any work
while it is open.

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
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

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

SEEK_SECONDS = 3
SPEEDS = (1.0, 0.75, 0.5, 0.25)

NOTES_FILENAME = "other_manual.json"

# Notes are saved to and loaded from <NOTES_ROOT>/<song name>/other_manual.json,
# where the song name is the MP3's file name without ".mp3". Relative to
# this script's folder.
NOTES_ROOT = Path(__file__).resolve().parent / "separated" / "htdemucs"

# If one of these stems is opened (e.g. separated/htdemucs/timeless/other.mp3),
# the song name is taken from its folder instead of the file name.
STEM_NAMES = ("bass", "drums", "other", "vocals")

# Opening this file asks for the song name instead (pre-filled with the
# file's folder name). Song names may use letters, numbers, _ and -.
OTHER_STEM_FILENAME = "other.mp3"
SONG_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
NOTES_FORMAT = "other_manual_v1"

# A notes file whose duration differs from the opened MP3 by more than this
# belongs to another song and is not loaded or overwritten.
DURATION_TOLERANCE_SECONDS = 0.5

DEFAULT_COLOR = (255, 120, 0)
DEFAULT_FADE = 40

# A key pressed within this many frames after the first key of a group is
# snapped onto that first key's frame, so chords land together.
# 0 disables snapping.
SNAP_FRAMES = 0

# Audio offset slider range and step (milliseconds of real time). Notes are
# placed this long before the key press, so the key can be pressed late.
MAX_AUDIO_OFFSET_MS = 999
AUDIO_OFFSET_STEP_MS = 10
DEFAULT_AUDIO_OFFSET_MS = 100

# Number of LEDs one key press lights, starting at the key's LED.
NOTE_WIDTHS = (1, 2, 3, 4, 5, 6)

# "Snap to Nearest Onset" moves each note to the frame with the highest
# overall onset within this many frames either side.
ONSET_SNAP_RADIUS_FRAMES = 1
ONSETS_FILENAME = "other_onsets.json"

# Smaller = lower audio latency, but may crackle on slow machines.
AUDIO_BUFFER_SAMPLES = 512

UPDATE_INTERVAL_MS = 20

# While the Advanced window is open, its note list is rebuilt this long
# after the notes change, once per burst of key presses.
ADVANCED_REFRESH_DELAY_MS = 400

# Vertical centre of the seek bar line, in its canvas.
SEEK_BAR_Y = 15

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


def song_name_for(song_path):
    """Song name from the MP3's file name, or its folder for a stem file."""
    song_name = song_path.stem

    if song_name.lower() in STEM_NAMES:
        song_name = song_path.parent.name

    return song_name


def notes_path_for(song_name):
    """separated/htdemucs/<song name>/other_manual.json."""
    return NOTES_ROOT / song_name / NOTES_FILENAME


def make_note(frame, position, rgb, fade):
    return {
        "frame": int(frame),
        "time": round(int(frame) / FRAME_RATE, 4),
        "position": int(position),
        "r": int(rgb[0]),
        "g": int(rgb[1]),
        "b": int(rgb[2]),
        "fade": int(fade),
    }


def text_colour_for(rgb):
    """Black or white text, whichever reads better on this background."""
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "#000000" if luminance > 140 else "#ffffff"


def ms_to_frames(milliseconds):
    """Nearest whole number of frames, rounding halves away from zero."""
    frames = math.floor(abs(milliseconds) * FRAME_RATE / 1000 + 0.5)
    return int(math.copysign(frames, milliseconds))


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

    def add(self, frame, positions, rgb, fade):
        """
        Add one note per LED position; undone together as one step.
        Returns the added notes.
        """
        return self.apply_change(
            [],
            [make_note(frame, position, rgb, fade) for position in positions],
        )

    def apply_change(self, removed, added):
        """
        Remove the `removed` notes and add the `added` ones as one undo step.
        An added note replaces any other note on the same frame and LED.
        Returns the notes actually added.
        """
        removed_ids = {id(note) for note in removed}
        remaining = [note for note in self.notes if id(note) not in removed_ids]

        # If two added notes share a frame and LED, the later one wins.
        by_key = {(note["frame"], note["position"]): note for note in added}
        added = list(by_key.values())

        replaced = [
            note for note in remaining
            if (note["frame"], note["position"]) in by_key
        ]
        replaced_ids = {id(note) for note in replaced}

        self.notes = [note for note in remaining if id(note) not in replaced_ids] + added
        self.undo_stack.append((list(removed) + replaced, added))
        self._changed()

        return added

    def undo(self):
        if not self.undo_stack:
            return False

        removed, added = self.undo_stack.pop()
        added_ids = {id(note) for note in added}

        self.notes = [note for note in self.notes if id(note) not in added_ids] + removed
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

        self.path.parent.mkdir(parents=True, exist_ok=True)

        # One note per line keeps the file readable, and building the whole
        # text first is far faster than json.dump(indent=2) on many notes.
        notes = data.pop("notes")
        header = json.dumps(data, indent=2)[:-2]

        if notes:
            note_lines = ",\n    ".join(json.dumps(note) for note in notes)
            text = f'{header},\n  "notes": [\n    {note_lines}\n  ]\n}}\n'
        else:
            text = f'{header},\n  "notes": []\n}}\n'

        # Write to a temporary file first so a crash can't corrupt the notes.
        temporary = self.path.with_suffix(".json.tmp")

        with temporary.open("w", encoding="utf-8") as f:
            f.write(text)

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


class AdvancedWindow:
    """
    Separate window for selecting and editing existing notes. It only
    rebuilds its note list while it is open.
    """

    COLOUR_MODES = ("any colour", "colour is", "colour is not")

    def __init__(self, app):
        self.app = app
        self.visible = True
        self.refresh_pending = False

        # Selected notes, by tree row id (str(id(note))).
        self.selected = set()
        self.notes_by_iid = {}
        self.colour_tags = set()

        self.time_from = tk.StringVar()
        self.time_to = tk.StringVar()
        self.led_from = tk.StringVar()
        self.led_to = tk.StringVar()
        self.fade_from = tk.StringVar()
        self.fade_to = tk.StringVar()
        self.colour_mode = tk.StringVar(value=self.COLOUR_MODES[0])
        self.filter_colour = None

        self.shift_ms = tk.StringVar(value="40")
        self.set_time = tk.StringVar()
        self.shift_leds = tk.StringVar(value="1")
        self.set_fade = tk.StringVar(value=str(DEFAULT_FADE))

        window = tk.Toplevel(app.root)
        window.title("Advanced - Notes")
        window.configure(bg=BACKGROUND)
        window.geometry("1040x800")
        window.minsize(900, 780)
        window.protocol("WM_DELETE_WINDOW", app.toggle_advanced)
        self.window = window

        self._style_tree()

        controls = tk.Frame(window, bg=PANEL, width=380)
        controls.pack(side="left", fill="y", padx=(16, 0), pady=16)
        controls.pack_propagate(False)

        self._build_selection(controls)
        self._build_edits(controls)
        self._build_list(window)

        self.shift_ms.trace_add("write", lambda *_args: self._update_shift_label())
        self._update_shift_label()

        self.refresh()

    # --- Layout -------------------------------------------------------------

    def _style_tree(self):
        style = ttk.Style(self.window)
        style.theme_use("clam")
        style.configure(
            "Notes.Treeview",
            background=PANEL,
            fieldbackground=PANEL,
            foreground=FOREGROUND,
            rowheight=22,
            borderwidth=0,
            font=FONT,
        )
        style.configure(
            "Notes.Treeview.Heading",
            background="#2c2c36",
            foreground=FOREGROUND,
            relief="flat",
            font=FONT_BOLD,
        )
        style.map(
            "Notes.Treeview",
            background=[("selected", "#ffffff")],
            foreground=[("selected", "#000000")],
        )

    def _entry(self, parent, variable, width=7):
        return tk.Entry(
            parent,
            textvariable=variable,
            width=width,
            font=FONT,
            bg="#2c2c36",
            fg=FOREGROUND,
            insertbackground=FOREGROUND,
            relief="flat",
        )

    def _range_row(self, parent, label, from_var, to_var, unit=""):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=14, pady=3)

        self.app._label(row, label, width=7, anchor="w").pack(side="left")
        self._entry(row, from_var).pack(side="left")
        self.app._label(row, " to ").pack(side="left")
        self._entry(row, to_var).pack(side="left")
        self.app._label(row, f" {unit}", fg=MUTED).pack(side="left")

        return row

    def _build_selection(self, parent):
        app = self.app
        app._label(parent, "SELECT NOTES", font=FONT_BOLD, fg=MUTED).pack(
            anchor="w", padx=14, pady=(14, 2)
        )
        app._label(parent, "Leave a box empty for no limit.", fg=MUTED).pack(
            anchor="w", padx=14, pady=(0, 4)
        )

        time_row = self._range_row(parent, "Time", self.time_from, self.time_to, "s")
        app._button(time_row, "Playhead", lambda: self._fill_playhead(self.time_from)).pack(side="left", padx=(6, 2))
        app._button(time_row, "Playhead", lambda: self._fill_playhead(self.time_to)).pack(side="left")

        self._range_row(parent, "LED", self.led_from, self.led_to, "(0-47)")
        self._range_row(parent, "Fade", self.fade_from, self.fade_to, "(0-255)")

        colour_row = tk.Frame(parent, bg=PANEL)
        colour_row.pack(fill="x", padx=14, pady=3)

        mode_menu = tk.OptionMenu(colour_row, self.colour_mode, *self.COLOUR_MODES)
        mode_menu.config(
            font=FONT, bg="#2c2c36", fg=FOREGROUND, activebackground="#3a3a46",
            activeforeground=FOREGROUND, relief="flat", highlightthickness=0,
            width=11, takefocus=0,
        )
        mode_menu.pack(side="left")

        self.colour_button = tk.Menubutton(
            colour_row, text="pick colour", font=FONT, bg="#2c2c36", fg=FOREGROUND,
            activebackground="#3a3a46", activeforeground=FOREGROUND,
            relief="flat", padx=8, pady=3, takefocus=0,
        )
        self.colour_menu = tk.Menu(self.colour_button, tearoff=False)
        self.colour_button.config(menu=self.colour_menu)
        self.colour_button.pack(side="left", padx=6)

        actions = tk.Frame(parent, bg=PANEL)
        actions.pack(fill="x", padx=14, pady=(8, 2))

        for text, action in (
            ("Select", "select"),
            ("Add", "add"),
            ("Remove", "remove"),
            ("Keep only", "keep"),
        ):
            app._button(actions, text, lambda a=action: self.apply_filter(a)).pack(side="left", padx=(0, 4))

        app._label(
            parent,
            "Select: matches only.  Add: add matches.\n"
            "Remove: deselect matches.  Keep only: deselect the rest.",
            fg=MUTED, justify="left",
        ).pack(anchor="w", padx=14)

        other = tk.Frame(parent, bg=PANEL)
        other.pack(fill="x", padx=14, pady=(6, 0))
        app._button(other, "Select all", self.select_all).pack(side="left", padx=(0, 4))
        app._button(other, "Clear selection", self.clear_selection).pack(side="left")

        self.count_label = app._label(parent, "", font=FONT_BOLD, fg=ACCENT)
        self.count_label.pack(anchor="w", padx=14, pady=(8, 0))

    def _edit_row(self, parent, label, variable, unit, button, command):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=14, pady=3)

        self.app._label(row, label, width=12, anchor="w").pack(side="left")
        self._entry(row, variable).pack(side="left")
        self.app._label(row, f" {unit}", fg=MUTED, width=4, anchor="w").pack(side="left")
        self.app._button(row, button, command).pack(side="left")

        return row

    def _build_edits(self, parent):
        app = self.app
        app._label(parent, "EDIT SELECTED", font=FONT_BOLD, fg=MUTED).pack(
            anchor="w", padx=14, pady=(18, 4)
        )

        self._edit_row(parent, "Shift time by", self.shift_ms, "ms", "Shift", self.shift_time)

        self.shift_label = app._label(parent, "", fg=MUTED)
        self.shift_label.pack(anchor="w", padx=14)

        self._edit_row(parent, "Set time to", self.set_time, "s", "Set", self.set_times)
        self._edit_row(parent, "Shift LEDs by", self.shift_leds, "", "Shift", self.shift_positions)
        self._edit_row(parent, "Set fade to", self.set_fade, "", "Set", self.set_fades)

        colour_row = tk.Frame(parent, bg=PANEL)
        colour_row.pack(fill="x", padx=14, pady=3)
        app._label(colour_row, "Set colour", width=12, anchor="w").pack(side="left")
        app._button(colour_row, "Choose...", self.set_colours).pack(side="left")

        bottom = tk.Frame(parent, bg=PANEL)
        bottom.pack(fill="x", padx=14, pady=(12, 0))
        app._button(bottom, "Delete selected", self.delete_selected).pack(side="left", padx=(0, 4))
        app._button(bottom, "Undo", app.undo).pack(side="left")

        self.status_label = app._label(parent, "", fg=MUTED, wraplength=340, justify="left")
        self.status_label.pack(anchor="w", padx=14, pady=(10, 0))

    def _build_list(self, parent):
        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(side="left", fill="both", expand=True, padx=12, pady=16)

        self.app._label(
            frame,
            "NOTES   (click, Ctrl+click and Shift+click to select; double-click to jump there)",
            font=FONT_BOLD, fg=MUTED,
        ).pack(anchor="w", padx=10, pady=(10, 4))

        columns = ("time", "frame", "led", "colour", "fade")
        tree = ttk.Treeview(
            frame, columns=columns, show="headings",
            selectmode="extended", style="Notes.Treeview",
        )

        for column, heading, width in (
            ("time", "Time (s)", 90),
            ("frame", "Frame", 70),
            ("led", "LED", 50),
            ("colour", "Colour", 100),
            ("fade", "Fade", 60),
        ):
            tree.heading(column, text=heading)
            tree.column(column, width=width, anchor="center")

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y", pady=(0, 10))
        tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))

        tree.bind("<<TreeviewSelect>>", self._tree_selected)
        tree.bind("<Double-1>", self._jump_to_note)

        self.tree = tree

    # --- Showing and refreshing --------------------------------------------

    def show(self):
        self.window.deiconify()
        self.window.lift()
        self.visible = True
        self.refresh()

    def hide(self):
        self.window.withdraw()
        self.visible = False

    def schedule_refresh(self):
        """Rebuild the list soon, at most once per burst of changes."""
        if not self.visible or self.refresh_pending:
            return

        self.refresh_pending = True
        self.window.after(ADVANCED_REFRESH_DELAY_MS, self.refresh)

    def refresh(self):
        self.refresh_pending = False

        if not self.visible:
            return

        notes = self.app.notes.notes if self.app.notes else []
        tree = self.tree

        tree.delete(*tree.get_children())
        self.notes_by_iid = {}

        for note in notes:
            rgb = (note["r"], note["g"], note["b"])
            colour = rgb_hex(rgb)
            tag = "c" + colour[1:]

            if tag not in self.colour_tags:
                tree.tag_configure(tag, background=colour, foreground=text_colour_for(rgb))
                self.colour_tags.add(tag)

            iid = str(id(note))
            self.notes_by_iid[iid] = note

            tree.insert(
                "", "end", iid=iid, tags=(tag,),
                values=(f"{note['time']:.2f}", note["frame"], note["position"], colour, note["fade"]),
            )

        self.selected &= set(self.notes_by_iid)
        tree.selection_set(list(self.selected))

        self._rebuild_colour_menu(notes)
        self._update_count()

    def _rebuild_colour_menu(self, notes):
        counts = {}

        for note in notes:
            rgb = (note["r"], note["g"], note["b"])
            counts[rgb] = counts.get(rgb, 0) + 1

        self.colour_menu.delete(0, "end")

        for rgb, count in sorted(counts.items(), key=lambda item: -item[1]):
            self.colour_menu.add_command(
                label=f"{rgb_hex(rgb)}   ({count} notes)",
                background=rgb_hex(rgb),
                foreground=text_colour_for(rgb),
                command=lambda c=rgb: self._choose_filter_colour(c),
            )

        if self.filter_colour is not None and self.filter_colour not in counts:
            self._choose_filter_colour(None)

    def _choose_filter_colour(self, rgb):
        self.filter_colour = rgb

        if rgb is None:
            self.colour_button.config(text="pick colour", bg="#2c2c36", fg=FOREGROUND)
        else:
            self.colour_button.config(
                text=rgb_hex(rgb), bg=rgb_hex(rgb), fg=text_colour_for(rgb),
                activebackground=rgb_hex(rgb),
            )

    def _update_count(self):
        total = len(self.notes_by_iid)
        self.count_label.config(text=f"{len(self.selected)} of {total} notes selected")

    def _update_shift_label(self):
        try:
            milliseconds = float(self.shift_ms.get())
        except ValueError:
            self.shift_label.config(text="Enter a number of milliseconds (negative = earlier).")
            return

        frames = ms_to_frames(milliseconds)
        actual = frames * 1000 / FRAME_RATE
        plural = "" if abs(frames) == 1 else "s"
        self.shift_label.config(
            text=f"= {frames:+d} frame{plural} ({actual:+.0f} ms). "
            f"Notes move in {1000 / FRAME_RATE:.0f} ms steps."
        )

    # --- Selection ----------------------------------------------------------

    def _tree_selected(self, _event):
        self.selected = set(self.tree.selection())
        self._update_count()

    def _set_selection(self, selected):
        self.selected = set(selected)
        self.tree.selection_set(list(self.selected))

        if self.selected:
            first = min(self.selected, key=lambda iid: self.tree.index(iid))
            self.tree.see(first)

        self._update_count()

    def select_all(self):
        self._set_selection(self.notes_by_iid)

    def clear_selection(self):
        self._set_selection(())

    def _fill_playhead(self, variable):
        variable.set(f"{self.app.player.position():.2f}")

    def _matcher(self):
        """A note -> bool test from the filter boxes. Raises ValueError."""

        def number(variable, name):
            text = variable.get().strip()

            if not text:
                return None

            try:
                return float(text)
            except ValueError:
                raise ValueError(f"{name} must be a number, not '{text}'.")

        time_from = number(self.time_from, "Time from")
        time_to = number(self.time_to, "Time to")
        led_from = number(self.led_from, "LED from")
        led_to = number(self.led_to, "LED to")
        fade_from = number(self.fade_from, "Fade from")
        fade_to = number(self.fade_to, "Fade to")

        # Compare times as frames so a typed time matches its frame exactly.
        frame_from = None if time_from is None else round(time_from * FRAME_RATE)
        frame_to = None if time_to is None else round(time_to * FRAME_RATE)

        mode = self.colour_mode.get()
        colour = self.filter_colour

        if mode != self.COLOUR_MODES[0] and colour is None:
            raise ValueError("Pick a colour for the colour filter, or set it to 'any colour'.")

        def within(value, low, high):
            return (low is None or value >= low) and (high is None or value <= high)

        def matches(note):
            if not within(note["frame"], frame_from, frame_to):
                return False
            if not within(note["position"], led_from, led_to):
                return False
            if not within(note["fade"], fade_from, fade_to):
                return False

            if mode != self.COLOUR_MODES[0]:
                same = (note["r"], note["g"], note["b"]) == colour

                if same != (mode == "colour is"):
                    return False

            return True

        return matches

    def apply_filter(self, action):
        try:
            matches_test = self._matcher()
        except ValueError as exc:
            messagebox.showerror("Filter", str(exc), parent=self.window)
            return

        matches = {
            iid for iid, note in self.notes_by_iid.items() if matches_test(note)
        }

        if action == "select":
            selected = matches
        elif action == "add":
            selected = self.selected | matches
        elif action == "remove":
            selected = self.selected - matches
        else:
            selected = self.selected & matches

        self._set_selection(selected)

    def _jump_to_note(self, event):
        iid = self.tree.identify_row(event.y)

        if iid in self.notes_by_iid:
            self.app.player.seek(self.notes_by_iid[iid]["time"])

    # --- Edits --------------------------------------------------------------

    def _edit(self, change, description):
        """Replace each selected note with change(copy) as one undo step."""
        if self.app.notes is None:
            return

        notes = [self.notes_by_iid[iid] for iid in self.selected if iid in self.notes_by_iid]

        if not notes:
            messagebox.showerror("Edit", "Select some notes first.", parent=self.window)
            return

        last_frame = max(0, int(self.app.player.duration * FRAME_RATE) - 1)
        new_notes = []

        for note in notes:
            new = change(dict(note))

            if new is None:
                continue

            new_notes.append(make_note(
                min(max(new["frame"], 0), last_frame),
                min(max(new["position"], 0), LED_COUNT - 1),
                (new["r"], new["g"], new["b"]),
                min(max(new["fade"], 0), 255),
            ))

        count_before = len(self.app.notes.notes)
        added = self.app.notes.apply_change(notes, new_notes)
        merged = count_before - len(self.app.notes.notes) - (len(notes) - len(new_notes))

        self.selected = {str(id(note)) for note in added}
        self.app._update_count()
        self.app._draw_note_ticks()
        self.refresh()

        text = f"{description} ({len(notes)} notes)."

        if merged > 0:
            text += (
                f" {merged} note{'s' if merged != 1 else ''} merged because "
                f"they ended up on the same frame and LED as another note."
            )

        self.status_label.config(text=text + " Ctrl+Z undoes this.")

    def _read_int(self, variable, name):
        try:
            return int(round(float(variable.get())))
        except ValueError:
            messagebox.showerror("Edit", f"{name} must be a number.", parent=self.window)
            return None

    def shift_time(self):
        try:
            milliseconds = float(self.shift_ms.get())
        except ValueError:
            messagebox.showerror("Edit", "Shift must be a number of milliseconds.", parent=self.window)
            return

        frames = ms_to_frames(milliseconds)

        if frames == 0:
            messagebox.showerror(
                "Edit",
                f"{milliseconds:g} ms rounds to 0 frames. Notes move in "
                f"{1000 / FRAME_RATE:.0f} ms steps.",
                parent=self.window,
            )
            return

        def change(note):
            note["frame"] += frames
            return note

        plural = "" if abs(frames) == 1 else "s"
        self._edit(change, f"Shifted by {frames:+d} frame{plural} ({frames * 1000 / FRAME_RATE:+.0f} ms)")

    def set_times(self):
        try:
            seconds = float(self.set_time.get())
        except ValueError:
            messagebox.showerror("Edit", "Time must be a number of seconds.", parent=self.window)
            return

        frame = round(seconds * FRAME_RATE)

        def change(note):
            note["frame"] = frame
            return note

        self._edit(change, f"Moved to {frame / FRAME_RATE:.2f} s (frame {frame})")

    def shift_positions(self):
        leds = self._read_int(self.shift_leds, "LED shift")

        if leds is None:
            return

        def change(note):
            note["position"] += leds
            return note

        self._edit(change, f"Shifted LEDs by {leds:+d}")

    def set_fades(self):
        fade = self._read_int(self.set_fade, "Fade")

        if fade is None:
            return

        def change(note):
            note["fade"] = fade
            return note

        self._edit(change, f"Set fade to {min(max(fade, 0), 255)}")

    def set_colours(self):
        result = colorchooser.askcolor(
            color=rgb_hex(self.app.color), title="New colour", parent=self.window
        )

        if not result or not result[0]:
            return

        rgb = tuple(int(round(c)) for c in result[0])

        def change(note):
            note["r"], note["g"], note["b"] = rgb
            return note

        self._edit(change, f"Set colour to {rgb_hex(rgb)}")

    def delete_selected(self):
        self._edit(lambda note: None, "Deleted")


class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.player = AudioPlayer()
        self.notes = None

        self.color = DEFAULT_COLOR
        self.held_keys = set()

        # Created the first time it is opened.
        self.advanced = None

        # Frame of the first key in the current snap group.
        self.snap_anchor_frame = None

        # Overall onset per frame from other_onsets.json, while snapping.
        self.onsets = None

        self.song_token = 0
        self.pending_speed = None
        self.preparing = set()
        self.results = queue.Queue()

        self.fade_var = tk.IntVar(value=DEFAULT_FADE)
        self.offset_var = tk.IntVar(value=DEFAULT_AUDIO_OFFSET_MS)
        self.width_var = tk.IntVar(value=1)
        self.onset_snap_var = tk.BooleanVar(value=False)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.window_var = tk.IntVar(value=0)

        root.title("Manual Other Recorder")
        root.configure(bg=BACKGROUND)
        root.geometry("1180x640")
        root.minsize(980, 640)

        self._build_header()
        self._build_body()
        self._bind_keys()

        self._draw_led_strip()
        self._update_fade_label()
        self._update_offset_label()
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

    def _choice_button(self, parent, text, value, variable, command):
        return tk.Radiobutton(
            parent,
            text=text,
            value=value,
            variable=variable,
            command=command,
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

        self.advanced_button = self._button(header, "Advanced  ▸", self.toggle_advanced)
        self.advanced_button.pack(side="right")

        self.status_label = self._label(header, "", fg=ACCENT)
        self.status_label.pack(side="right", padx=12)

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
            self._choice_button(
                speed_row, f"{speed:g}x", speed, self.speed_var, self.choose_speed
            ).pack(side="left", padx=(0, 4))

        self._label(sidebar, "NOTE WIDTH (LEDs)", font=FONT_BOLD, fg=MUTED).pack(pady=(16, 2), **pad)

        width_row = tk.Frame(sidebar, bg=PANEL)
        width_row.pack(fill="x", **pad)

        for width in NOTE_WIDTHS:
            self._choice_button(
                width_row, str(width), width, self.width_var, self._draw_led_strip
            ).pack(side="left", padx=(0, 4))

        tk.Checkbutton(
            sidebar,
            text="Snap to Nearest Onset",
            variable=self.onset_snap_var,
            command=self.toggle_onset_snap,
            font=FONT_BOLD,
            fg=FOREGROUND,
            bg=PANEL,
            selectcolor="#2c2c36",
            activebackground=PANEL,
            activeforeground=FOREGROUND,
            highlightthickness=0,
            takefocus=0,
        ).pack(pady=(12, 0), **pad)

        self._label(sidebar, "AUDIO OFFSET", font=FONT_BOLD, fg=MUTED).pack(pady=(16, 0), **pad)

        offset_scale = self._scale(
            sidebar, self.offset_var, 0, MAX_AUDIO_OFFSET_MS,
            lambda _value: self._update_offset_label(), 220,
        )
        offset_scale.config(resolution=AUDIO_OFFSET_STEP_MS)
        offset_scale.pack(**pad)

        self.offset_label = self._label(sidebar, "", fg=MUTED)
        self.offset_label.pack(**pad)

        self._label(sidebar, "NOTES", font=FONT_BOLD, fg=MUTED).pack(pady=(16, 2), **pad)

        self.count_label = self._label(sidebar, "0 notes")
        self.count_label.pack(**pad)

        self._button(sidebar, "Undo last note", self.undo).pack(pady=(6, 14), **pad)

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

        # Note ticks sit on their own canvas so the moving playhead never
        # forces them to redraw; with many notes that redraw was the main
        # cause of lag later in the song.
        self.tick_canvas = tk.Canvas(parent, height=12, bg=PANEL, highlightthickness=0)
        self.tick_canvas.pack(fill="x", padx=16, pady=(10, 0))
        self.tick_canvas.bind("<Configure>", lambda _event: self._draw_note_ticks())

        self.seek_canvas = tk.Canvas(parent, height=30, bg=PANEL, highlightthickness=0)
        self.seek_canvas.pack(fill="x", padx=16, pady=(0, 14))
        self.seek_canvas.bind("<Configure>", lambda _event: self._draw_seek_static())

        for canvas in (self.tick_canvas, self.seek_canvas):
            canvas.bind("<Button-1>", self._seek_click)
            canvas.bind("<B1-Motion>", self._seek_click)

        self._draw_seek_static()

        self._label(
            parent,
            "Keys 1-9, 0: place note     Space: play / pause     "
            "Left / Right: -5 s / +5 s     R: restart     Ctrl+Z: undo",
            fg=MUTED,
        ).pack(side="bottom", anchor="w", padx=16, pady=14)

    # --- Keys ---------------------------------------------------------------

    def _bind_keys(self):
        # Stop buttons, radio buttons and sliders from reacting to these
        # keys, so they always control playback.
        for widget_class in ("Button", "Radiobutton", "Checkbutton", "Scale"):
            for sequence in ("<space>", "<Left>", "<Right>", "<Up>", "<Down>"):
                self.root.unbind_class(widget_class, sequence)

        self.root.bind_all("<KeyPress>", self._key_press)
        self.root.bind_all("<KeyRelease>", self._key_release)
        self.root.bind_all("<Control-z>", lambda _event: self.undo())

    def _key_press(self, event):
        # Let text boxes (Advanced window) receive typing normally.
        if isinstance(event.widget, tk.Entry):
            return None

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

        if not path:
            return

        path = Path(path)
        song_name = None

        # other.mp3 doesn't say which song it is, so ask.
        if path.name.lower() == OTHER_STEM_FILENAME:
            song_name = self.ask_song_name(path)

            if song_name is None:
                return

        self.load_song(path, song_name)

    def ask_song_name(self, path):
        """Ask which song an other.mp3 belongs to. None if cancelled."""
        suggestion = path.parent.name

        while True:
            name = simpledialog.askstring(
                "Song name",
                f"Which song is this {path.name} from?\n"
                f"(e.g. worry_child, timeless, rebel_city)",
                initialvalue=suggestion,
                parent=self.root,
            )

            if name is None:
                return None

            name = name.strip()

            if not SONG_NAME_PATTERN.fullmatch(name):
                messagebox.showerror(
                    "Song name",
                    "Use only letters, numbers, underscores and hyphens, "
                    "e.g. worry_child.",
                )
                suggestion = name
                continue

            if not (NOTES_ROOT / name).is_dir() and not messagebox.askyesno(
                "Song name",
                f"There is no song folder called '{name}' in\n{NOTES_ROOT}.\n\n"
                f"Use this name anyway? (The folder is created when you "
                f"place the first note.)",
            ):
                suggestion = name
                continue

            return name

    def load_song(self, path, song_name=None):
        if song_name is None:
            song_name = song_name_for(path)

        self.status_label.config(text="Loading...")
        self.root.update_idletasks()

        try:
            self.player.load(path)
        except Exception as exc:
            self.status_label.config(text="")
            messagebox.showerror("Could not open song", str(exc))
            return

        notes = NotesFile(notes_path_for(song_name), path.name, self.player.duration)

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

        self.song_label.config(
            text=path.name if path.stem == song_name else f"{song_name}  ({path.name})"
        )
        self.file_label.config(text=f"Notes: {notes.path}")
        self.status_label.config(text="")

        self._update_count()
        self._draw_seek_static()

        if self.advanced is not None:
            self.advanced.selected.clear()
            self.advanced.schedule_refresh()

        # Reload onsets for the new song, or untick if it has none.
        self.onsets = None

        if self.onset_snap_var.get():
            self.toggle_onset_snap()

    def toggle_advanced(self):
        if self.advanced is None:
            self.advanced = AdvancedWindow(self)
        elif self.advanced.visible:
            self.advanced.hide()
        else:
            self.advanced.show()

        arrow = "▾" if self.advanced.visible else "▸"
        self.advanced_button.config(text=f"Advanced  {arrow}")

    def notes_changed(self, redraw_all=False):
        """Update everything that shows the notes after they change."""
        self._update_count()

        if redraw_all:
            self._draw_note_ticks()

        if self.advanced is not None:
            self.advanced.schedule_refresh()

    def toggle_onset_snap(self):
        """Load the song's onsets when ticked; untick with an error if missing."""
        if not self.onset_snap_var.get():
            self.onsets = None
            return

        if self.notes is None:
            self.onset_snap_var.set(False)
            messagebox.showerror("Snap to Nearest Onset", "Open a song first.")
            return

        path = self.notes.path.parent / ONSETS_FILENAME

        try:
            with path.open("r", encoding="utf-8") as f:
                frames = json.load(f)["frames"]

            onsets = np.array([float(frame["onset"]) for frame in frames])
        except FileNotFoundError:
            self._onset_snap_failed(f"{path} does not exist.")
            return
        except Exception as exc:
            self._onset_snap_failed(f"Could not read {path}:\n{exc}")
            return

        song_frames = self.player.duration * FRAME_RATE

        if abs(len(onsets) - song_frames) > DURATION_TOLERANCE_SECONDS * FRAME_RATE:
            self._onset_snap_failed(
                f"{path} has {len(onsets)} frames, but this song has about "
                f"{song_frames:.0f}. It probably belongs to another song."
            )
            return

        self.onsets = onsets

    def _onset_snap_failed(self, message):
        self.onsets = None
        self.onset_snap_var.set(False)
        messagebox.showerror("Snap to Nearest Onset", message)

    def snap_to_onset(self, frame):
        """Frame with the highest onset within ONSET_SNAP_RADIUS_FRAMES."""
        if self.onsets is None or frame >= len(self.onsets):
            return frame

        start = max(0, frame - ONSET_SNAP_RADIUS_FRAMES)
        end = min(len(self.onsets), frame + ONSET_SNAP_RADIUS_FRAMES + 1)
        window = self.onsets[start:end]

        # On a tie, keep the frame the key was pressed on.
        if window.max() <= self.onsets[frame]:
            return frame

        return start + int(np.argmax(window))

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

        # The offset is real time, so at slower speeds it covers less song.
        offset_seconds = self.offset_var.get() / 1000.0 * self.player.speed
        song_time = max(0.0, self.player.position() - offset_seconds)

        last_frame = max(0, int(self.player.duration * FRAME_RATE) - 1)
        frame = min(last_frame, int(round(song_time * FRAME_RATE)))

        if self.onset_snap_var.get():
            frame = self.snap_to_onset(frame)

        # Snap onto the first key of a group pressed within SNAP_FRAMES.
        anchor = self.snap_anchor_frame

        if anchor is not None and 0 <= frame - anchor <= SNAP_FRAMES:
            frame = anchor
        else:
            self.snap_anchor_frame = frame

        # The key's LED and the LEDs after it, stopping at the last LED.
        first = self.window_var.get() + key_index
        positions = range(first, min(LED_COUNT, first + self.width_var.get()))

        added = self.notes.add(frame, positions, self.color, int(self.fade_var.get()))
        self._add_note_ticks(added)
        self.notes_changed()

    def undo(self):
        if self.notes is not None and self.notes.undo():
            self.notes_changed(redraw_all=True)

        return "break"

    # --- Drawing ------------------------------------------------------------

    def _update_color_label(self):
        self.color_label.config(text="RGB {}, {}, {}".format(*self.color))

    def _update_fade_label(self):
        fade = int(self.fade_var.get())
        self.fade_label.config(text=f"{fade}  ({fade_description(fade)})")

    def _update_offset_label(self):
        offset = int(self.offset_var.get())

        if offset == 0:
            self.offset_label.config(text="0 ms  (notes land on the key press)")
        else:
            self.offset_label.config(text=f"{offset} ms  (press {offset} ms after the note)")

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
        self.led_colours = [LED_OFF] * LED_COUNT

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
        y = SEEK_BAR_Y

        canvas.create_line(self.seek_left, y, self.seek_right, y, fill=TRACK, width=6, capstyle="round")

        self.seek_progress = canvas.create_line(
            self.seek_left, y, self.seek_left, y, fill=ACCENT, width=6, capstyle="round"
        )
        self.seek_head = canvas.create_oval(0, 0, 0, 0, fill=FOREGROUND, outline="")

        self._draw_note_ticks()

    def _tick_x(self, note):
        canvas_width = max(self.tick_canvas.winfo_width(), 200)
        span = canvas_width - 12
        return round(6 + note["time"] / self.player.duration * span)

    def _draw_note_ticks(self):
        """One tick per pixel column that has notes, in the latest note's colour."""
        canvas = self.tick_canvas
        canvas.delete("all")

        if self.notes is None or self.player.duration <= 0:
            return

        # Many notes share a pixel (chords, wide notes), so draw each pixel
        # once instead of one item per note.
        columns = {}

        for note in self.notes.notes:
            columns[self._tick_x(note)] = (note["r"], note["g"], note["b"])

        for x, rgb in columns.items():
            canvas.create_line(x, 1, x, 11, fill=rgb_hex(rgb), width=2)

    def _add_note_ticks(self, notes):
        """Ticks for newly placed notes, without redrawing the others."""
        if self.player.duration <= 0:
            return

        for note in notes:
            x = self._tick_x(note)
            self.tick_canvas.create_line(
                x, 1, x, 11,
                fill=rgb_hex((note["r"], note["g"], note["b"])),
                width=2,
            )

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
        self.seek_canvas.coords(self.seek_progress, self.seek_left, SEEK_BAR_Y, x, SEEK_BAR_Y)
        self.seek_canvas.coords(self.seek_head, x - 7, SEEK_BAR_Y - 7, x + 7, SEEK_BAR_Y + 7)

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

            # Only redraw LEDs whose colour changed.
            if self.led_colours[index] != colour:
                self.led_colours[index] = colour
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
