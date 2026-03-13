# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""from generate_piano_urdf.py"""
WHITE_KEY_LENGTH = 0.15
WHITE_KEY_WIDTH = 0.022
WHITE_KEY_HEIGHT = 0.015
WHITE_KEY_JOINT_MAX_ANGLE = 0.3
WHITE_KEY_MASS = 0.05

BLACK_KEY_LENGTH = 0.09
BLACK_KEY_WIDTH = 0.015
BLACK_KEY_HEIGHT = 0.01
BLACK_KEY_JOINT_MAX_ANGLE = 0.25
BLACK_KEY_MASS = 0.03

BASE_SIZE = [0.3, 1.5, 0.05]
BASE_POS = [0, 0, 0.025]
WHITE_KEY_X_OFFSET = 0.01
WHITE_KEY_Z_OFFSET = 0.01
BLACK_KEY_X_OFFSET = 0.015
BLACK_KEY_Z_OFFSET = 0.015
SPACING = 0.002
NUM_WHITE_KEYS = 52

WHITE_KEY_INDICES = [
    0,2,3,5,7,8,10,12,14,15,17,19,20,22,24,26,27,29,31,32,34,36,
    38,39,41,43,44,46,48,50,51,53,55,56,58,60,62,63,65,67,68,70,
    72,74,75,77,79,80,82,84,86,87
]
BLACK_TWIN_KEY_INDICES = [
    4,6,16,18,28,30,40,42,52,54,64,66,76,78
]
BLACK_TRIPLET_KEY_INDICES = [
    1,9,11,13,21,23,25,33,35,37,45,47,49,57,59,61,69,71,73,81,83,85
]


"""Music constants."""

MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127

# MIDI pitch number of the lowest note on the piano (A0).
MIN_MIDI_PITCH_PIANO = 21
# MIDI pitch number of the highest note on the piano (C8).
MAX_MIDI_PITCH_PIANO = 108

# Min and max key numbers on the piano.
MIN_KEY_NUMBER = 0
MAX_KEY_NUMBER = 87
NUM_KEYS = MAX_KEY_NUMBER - MIN_KEY_NUMBER + 1

# Notes in an octave.
NOTES_IN_OCTAVE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
assert len(NOTES_IN_OCTAVE) == 12

# Notes on an 88-key piano from left (A0) to right (C8).
NOTES = []
NOTES.extend(["A0", "A#0", "B0"])
for octave in range(1, 8):
    for note in NOTES_IN_OCTAVE:
        NOTES.append(note + str(octave))
NOTES.append("C8")
assert len(NOTES) == NUM_KEYS

# Mapping for converting between a key number and its note name.
KEY_NUMBER_TO_NOTE_NAME = {i: note for i, note in enumerate(NOTES)}
NOTE_NAME_TO_KEY_NUMBER = {note: i for i, note in enumerate(NOTES)}

# Mapping for converting between a note name and its MIDI pitch number.
MIDI_NUMBER_TO_NOTE_NAME = {i + 21: name for i, name in enumerate(NOTES)}
NOTE_NAME_TO_MIDI_NUMBER = {v: k for k, v in MIDI_NUMBER_TO_NOTE_NAME.items()}

# Sampling frequency of the audio, in Hz.
SAMPLING_RATE = 44100

SUSTAIN_PEDAL_CC_NUMBER = 64
MIN_CC_VALUE = 0
MAX_CC_VALUE = 127

MIN_VELOCITY = 0
MAX_VELOCITY = 127

# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

__version__ = "1.0.10"

# Path to the root of the project.
_PROJECT_ROOT = Path(__file__).parent.parent

# Path to the soundfont directory.
_SOUNDFONT_PATH = _PROJECT_ROOT / "music" / "soundfonts"

# TimGM6mb.sf2 is the default soundfont file that is packaged with the pip install or
# available when installing from source using `bash scripts/install_deps.sh`.
_DEFAULT_SF2_PATH = _SOUNDFONT_PATH / "TimGM6mb.sf2"

# We first check if the user has a .robopianistrc file in their home directory. If so,
# we check if it specifies a soundfont file. If so, we use that.
_RC_FILE = Path.home() / ".robopianistrc"
if _RC_FILE.exists():
    found = False
    with open(_RC_FILE, "r") as f:
        for line in f:
            if line.startswith("DEFAULT_SOUNDFONT="):
                soundfont_path = line.split("=")[1].strip()
                SF2_PATH = _SOUNDFONT_PATH / f"{soundfont_path}.sf2"
                if not SF2_PATH.exists():
                    SF2_PATH = _DEFAULT_SF2_PATH
                found = True
                break
    if not found:
        SF2_PATH = _DEFAULT_SF2_PATH
# Otherwise, we look in the soundfont directory. Our preference is for the higher
# quality SalamanderGrandPiano.sf2, but if that is not found, we fall back to the
# default soundfont file.
else:
    _SALAMANDER_SF2_PATH = _SOUNDFONT_PATH / "SalamanderGrandPiano.sf2"
    if _SALAMANDER_SF2_PATH.exists():
        SF2_PATH = _SALAMANDER_SF2_PATH
    else:
        if not _DEFAULT_SF2_PATH.exists():
            raise FileNotFoundError(
                f"The default soundfont file {_DEFAULT_SF2_PATH} does not exist. Make "
                "sure you have first run `bash scripts/install_deps.sh` in the root of "
                "the project directory."
            )
        SF2_PATH = _DEFAULT_SF2_PATH



# Wrist offset constants
DY_THUMB_L = -0.125
DY_INDEX_L = -0.0266
DY_MIDDLE_L = -0.0021
DY_RING_L = 0.0224
DY_PINKY_L = 0.0479
DY_FINGERS_L = [DY_THUMB_L, DY_INDEX_L, DY_MIDDLE_L, DY_RING_L, DY_PINKY_L]

DY_THUMB_R = 0.125
DY_INDEX_R = 0.0274
DY_MIDDLE_R = 0.0029
DY_RING_R = -0.0216
DY_PINKY_R = -0.0471
DY_FINGERS_R = [DY_THUMB_R, DY_INDEX_R, DY_MIDDLE_R, DY_RING_R, DY_PINKY_R]

DX_THUMB = -0.187
DX_INDEX = -0.004
DX_MIDDLE = 0.0
DX_RING = -0.008
DX_PINKY = -0.037
DX_FINGERS = [DX_THUMB, DX_INDEX, DX_MIDDLE, DX_RING, DX_PINKY]

COLORS = [
    [1.0, 0.0, 0.0, 1.0],  # red
    [1.0, 0.5, 0.0, 1.0],  # orange
    [1.0, 1.0, 0.0, 1.0],  # yellow
    [0.0, 1.0, 0.0, 1.0],  # green
    [0.0, 0.0, 1.0, 1.0],  # blue
]

def is_white_key(key):
    return key in WHITE_KEY_INDICES

__all__ = [
    "__version__",
    "SF2_PATH",
    "DY_FINGERS_L",
    "DY_FINGERS_R",
    "DX_FINGERS",
    "COLORS",
    "WHITE_KEY_INDICES",
    "is_white_key",
]

