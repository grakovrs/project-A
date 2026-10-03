import numpy as np
import json
import os
import soundfile as sf
import librosa

# ── CONFIG ─────────────────────────────────────────
BASE_DIR     = os.path.expanduser("~/Documents/advids")
TRACKS_DIR   = os.path.join(BASE_DIR, "tracks")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
TESTS_DIR    = os.path.join(BASE_DIR, "tests")

ANALYSIS = os.path.join(ANALYSIS_DIR, "energetic_01_analysis.json")
TRACK    = os.path.join(TRACKS_DIR,   "energetic_01.mp3")
OUTPUT   = os.path.join(TESTS_DIR,    "click_track.wav")

# Load analysis
with open(ANALYSIS) as f:
    data = json.load(f)

downbeats  = data["downbeats"]
beat_times = data["beat_times"]
duration   = data["duration"]

print(f"\nDownbeats detected ({len(downbeats)} total):")
for i, d in enumerate(downbeats):
    print(f"  Bar {i+1}: {d}s")

# Load audio
print(f"\nLoading audio...")
y, sr = librosa.load(TRACK, sr=44100)

# Generate click sounds
# Loud click for downbeats, soft click for other beats
def make_click(sr, freq=1000, duration_ms=30, amplitude=0.8):
    t = np.linspace(0, duration_ms/1000, int(sr * duration_ms/1000))
    click = amplitude * np.sin(2 * np.pi * freq * t)
    # Apply fast fade out
    fade = np.exp(-t * 80)
    return click * fade

def make_soft_click(sr, freq=600, duration_ms=20, amplitude=0.3):
    t = np.linspace(0, duration_ms/1000, int(sr * duration_ms/1000))
    click = amplitude * np.sin(2 * np.pi * freq * t)
    fade = np.exp(-t * 100)
    return click * fade

loud_click = make_click(sr)
soft_click = make_soft_click(sr)

# Mix clicks into audio
y_mix = y.copy()

# Add soft clicks at all beats
for bt in beat_times:
    start = int(bt * sr)
    end   = start + len(soft_click)
    if end < len(y_mix):
        y_mix[start:end] += soft_click

# Add loud clicks at downbeats (overwrites soft click)
for dt in downbeats:
    start = int(dt * sr)
    end   = start + len(loud_click)
    if end < len(y_mix):
        y_mix[start:end] += loud_click

# Normalize to prevent clipping
y_mix = y_mix / np.max(np.abs(y_mix)) * 0.95

# Save
sf.write(OUTPUT, y_mix, sr)
print(f"\n✓ Click track saved: click_track.wav")
print(f"\nListen carefully:")
print(f"  LOUD click = bar downbeat (beat 1)")
print(f"  soft click = other beats")
print(f"\nDoes every LOUD click land exactly where")
print(f"you expect the bar to start?\n")