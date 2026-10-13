import librosa
import madmom
import numpy as np
import json
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.expanduser("~/Documents/advids")
TRACKS_DIR   = os.path.join(BASE_DIR, "tracks")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

TRACK  = os.path.join(TRACKS_DIR,   "energetic_01.mp3")
OUTPUT = os.path.join(ANALYSIS_DIR, "energetic_01_analysis.json")

# Minimum onset threshold — points below this are treated as silence
MIN_ONSET_THRESHOLD = 3.0   # 0-100 normalized

# Attack time tolerance — window to match beat position with RMS peak
ATTACK_TOLERANCE = 0.05     # 50ms

print("\n" + "═"*50)
print("  AdVids — Deep Track Analysis v1.3")
print("═"*50)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load audio
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading audio...")
y, sr = librosa.load(TRACK)
duration = librosa.get_duration(y=y, sr=sr)
print(f"    Duration    : {round(duration, 2)}s")
print(f"    Sample rate : {sr}Hz")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Detect tempo + beats (Madmom)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Detecting tempo and beats (Madmom)...")
proc       = madmom.features.beats.RNNBeatProcessor()
beat_times = madmom.features.beats.BeatTrackingProcessor(fps=100)(proc(TRACK))
beat_times = [round(float(b), 3) for b in beat_times]

intervals     = np.diff(beat_times)
bpm           = round(float(60.0 / np.median(intervals)), 1)
beat_duration = round(float(np.median(intervals)), 4)
print(f"    BPM          : {bpm}")
print(f"    Beats found  : {len(beat_times)}")
print(f"    Beat duration: {beat_duration}s")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Detect downbeats + bar boundaries (Madmom)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Detecting downbeats and bars (Madmom)...")
dproc  = madmom.features.downbeats.RNNDownBeatProcessor()
dbproc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
    beats_per_bar=[3, 4], fps=100
)
downbeat_data = dbproc(dproc(TRACK))

downbeats = []
for entry in downbeat_data:
    time     = round(float(entry[0]), 3)
    position = int(entry[1])
    if position == 1:
        downbeats.append(time)

# Build bars from consecutive downbeats
bars = []
for i in range(len(downbeats)):
    bar_start = downbeats[i]
    bar_end   = downbeats[i+1] if i+1 < len(downbeats) else round(duration, 3)
    bars.append({
        "bar":      i + 1,
        "start":    bar_start,
        "end":      bar_end,
        "duration": round(bar_end - bar_start, 3)
    })

# Detect time signature from beats per bar
beats_per_bar_list = []
for i in range(len(downbeats) - 1):
    beats_in_bar = sum(
        1 for b in beat_times
        if downbeats[i] <= b < downbeats[i+1]
    )
    if beats_in_bar > 0:
        beats_per_bar_list.append(beats_in_bar)

time_signature = int(np.median(beats_per_bar_list)) if beats_per_bar_list else 4
bar1_end       = bars[0]["end"] if bars else 0

print(f"    Time signature : {time_signature}/4")
print(f"    Bars detected  : {len(bars)}")
print(f"    Bar 1 ends at  : {bar1_end}s (excluded from cut candidates)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Energy curve (Librosa)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Analyzing energy curve...")
rms       = librosa.feature.rms(y=y)[0]
rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

rms_min  = float(rms.min())
rms_max  = float(rms.max())
rms_mean = float(rms.mean())
rms_normalized = [
    (float(v) - rms_min) / (rms_max - rms_min) * 100
    for v in rms
]

# Climax — highest sustained energy (smoothed window)
window      = 20
smoothed    = np.convolve(rms_normalized, np.ones(window)/window, mode='same')
climax_idx  = int(np.argmax(smoothed))
climax_time = round(float(rms_times[climax_idx]), 3)
print(f"    Energy range : {round(rms_min,4)} → {round(rms_max,4)}")
print(f"    Climax at    : {climax_time}s "
      f"({round(climax_time/duration*100)}% through track)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Onset strength (Librosa)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Calculating onset strength...")
onset_env   = librosa.onset.onset_strength(y=y, sr=sr)
onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
onset_min   = float(onset_env.min())
onset_max   = float(onset_env.max())
print(f"    Onset range  : {round(onset_min,4)} → {round(onset_max,4)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def get_onset_strength(t):
    """Normalized onset strength (0-100) at time t."""
    idx = int(np.argmin(np.abs(onset_times - t)))
    raw = float(onset_env[idx])
    return round((raw - onset_min) / (onset_max - onset_min) * 100, 2)

def get_energy_at(t):
    """Normalized RMS energy (0-100) at time t."""
    idx = int(np.argmin(np.abs(rms_times - t)))
    return round(rms_normalized[idx], 2)

def is_downbeat(t, downbeats, tolerance=0.05):
    """True only if time t matches a detected downbeat within 50ms."""
    return any(abs(t - dt) < tolerance for dt in downbeats)

def get_bar_position_weight(t, downbeats, beat_times, time_sig,
                             onset_at_t, climax_t, duration):
    """
    Bar position weight based on beat position within bar.
    Downbeat = 100 only if onset strong (>40) OR near climax.
    Plain weak-onset downbeat = 40.
    Beat 3 in 4/4 = 50. Weak beats = 25. Subdivisions = 10.
    """
    bar_idx = None
    for i in range(len(downbeats)):
        bar_end = downbeats[i+1] if i+1 < len(downbeats) else float('inf')
        if downbeats[i] <= t < bar_end:
            bar_idx = i
            break
    if bar_idx is None:
        return 10

    bar_beats = [
        b for b in beat_times
        if downbeats[bar_idx] <= b < (
            downbeats[bar_idx+1] if bar_idx+1 < len(downbeats) else float('inf')
        )
    ]
    if not bar_beats:
        return 10

    near_climax = abs(t - climax_t) < (duration * 0.1)

    for beat_pos, bt in enumerate(bar_beats):
        if abs(t - bt) < 0.05:
            if beat_pos == 0:
                if onset_at_t > 40 or near_climax:
                    return 100
                else:
                    return 40
            elif beat_pos == 2 and time_sig == 4:
                return 50
            else:
                return 25
    return 10

def get_climax_bonus(t, climax_t, duration):
    """
    Climax proximity bonus — max 50 points.
    Active within 30% of track duration from climax.
    Falls off with distance.
    """
    distance     = abs(t - climax_t)
    max_distance = duration * 0.3
    if distance > max_distance:
        return 0.0
    bonus = (1 - (distance / max_distance)) * 50
    return round(bonus, 2)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Generate subdivisions with combined approach (bar 2 onwards)
# ─────────────────────────────────────────────────────────────────────────────
# Combined approach for secondary points (between downbeats):
#   HIGH   — beat position AND RMS/onset peak coincide (within ATTACK_TOLERANCE)
#   MEDIUM — beat position exists, no RMS peak nearby
#   LOW    — RMS/onset peak exists, no beat position nearby
#
# PRIMARY points (downbeats) always included — structurally determined by Madmom.
# HIGH + MEDIUM used for effect placement. LOW kept in JSON for analysis only.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Generating subdivisions with combined approach (bar 2 onwards)...")

eighth_duration    = beat_duration / 2
sixteenth_duration = beat_duration / 4

def find_rms_peaks_between(t_start, t_end):
    """
    Find local onset peaks between t_start and t_end.
    A peak is a point where onset_strength is locally maximum
    and above MIN_ONSET_THRESHOLD.
    Returns list of peak times.
    """
    mask = (onset_times >= t_start) & (onset_times < t_end)
    if not np.any(mask):
        return []

    local_times  = onset_times[mask]
    local_onsets = onset_env[mask]

    if len(local_onsets) < 3:
        return []

    peaks = []
    for i in range(1, len(local_onsets) - 1):
        if (local_onsets[i] > local_onsets[i-1] and
                local_onsets[i] > local_onsets[i+1]):
            raw_onset  = float(local_onsets[i])
            normalized = (raw_onset - onset_min) / (onset_max - onset_min) * 100
            if normalized >= MIN_ONSET_THRESHOLD:
                peaks.append(round(float(local_times[i]), 3))
    return peaks


subdivisions = []

for i, db in enumerate(downbeats):
    # Skip bar 1
    if db < bar1_end:
        continue

    next_db = downbeats[i+1] if i+1 < len(downbeats) else duration

    # Always add the downbeat itself (PRIMARY)
    subdivisions.append({
        "time":       db,
        "type":       "quarter",
        "confidence": "primary"
    })

    # Collect all mathematical beat positions between this downbeat and next
    beat_positions = []
    for bt in beat_times:
        if db < bt < next_db:
            beat_positions.append(round(bt, 3))
        # Eighth and sixteenth subdivisions anchored to every beat in range
        if db <= bt < next_db:
            eighth = round(bt + eighth_duration, 3)
            if db < eighth < next_db:
                beat_positions.append(eighth)
            s1 = round(bt + sixteenth_duration, 3)
            s2 = round(bt + sixteenth_duration * 3, 3)
            if db < s1 < next_db:
                beat_positions.append(s1)
            if db < s2 < next_db:
                beat_positions.append(s2)

    beat_positions = sorted(set(beat_positions))

    # Find actual RMS/onset peaks between downbeats
    rms_peaks = find_rms_peaks_between(db, next_db)
    used_peaks = set()

    # Assign HIGH or MEDIUM to each beat position
    for bp in beat_positions:
        if bp >= duration:
            continue

        onset_at_bp = get_onset_strength(bp)

        # Check if any RMS peak coincides within ATTACK_TOLERANCE
        matching_peak = None
        for pk in rms_peaks:
            if abs(bp - pk) <= ATTACK_TOLERANCE:
                matching_peak = pk
                break

        # Skip if no audio content and no RMS peak nearby
        if onset_at_bp < MIN_ONSET_THRESHOLD and matching_peak is None:
            continue

        confidence = "high" if matching_peak is not None else "medium"
        if matching_peak is not None:
            used_peaks.add(matching_peak)

        # Determine subdivision type
        sub_type = "sixteenth"
        for bt in beat_times:
            if abs(bp - bt) < 0.02:
                sub_type = "quarter"
                break
            elif abs(bp - (bt + eighth_duration)) < 0.02:
                sub_type = "eighth"
                break

        subdivisions.append({
            "time":       bp,
            "type":       sub_type,
            "confidence": confidence
        })

    # Add LOW confidence for unmatched RMS peaks
    for pk in rms_peaks:
        if pk in used_peaks or pk >= duration:
            continue
        subdivisions.append({
            "time":       pk,
            "type":       "rms_peak",
            "confidence": "low"
        })

subdivisions.sort(key=lambda x: x["time"])

high_count    = sum(1 for s in subdivisions if s["confidence"] == "high")
medium_count  = sum(1 for s in subdivisions if s["confidence"] == "medium")
low_count     = sum(1 for s in subdivisions if s["confidence"] == "low")
primary_count = sum(1 for s in subdivisions if s["confidence"] == "primary")

print(f"    Total subdivision points : {len(subdivisions)}")
print(f"      Primary  (downbeats)   : {primary_count}")
print(f"      HIGH     (beat+peak)   : {high_count}")
print(f"      MEDIUM   (beat only)   : {medium_count}")
print(f"      LOW      (peak only)   : {low_count}")
print(f"    Attack tolerance         : {ATTACK_TOLERANCE*1000:.0f}ms")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Score all subdivision points
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Scoring accent points...")

scored_points = []
for sub in subdivisions:
    t          = sub["time"]
    sub_type   = sub["type"]
    confidence = sub["confidence"]

    onset        = get_onset_strength(t)
    energy       = get_energy_at(t)
    bar_wt       = get_bar_position_weight(
                       t, downbeats, beat_times, time_signature,
                       onset, climax_time, duration)
    climax_bonus = get_climax_bonus(t, climax_time, duration)

    score = round(
        (onset        * 0.35) +
        (energy       * 0.20) +
        (bar_wt       * 0.25) +
        (climax_bonus * 0.20),
        2
    )

    scored_points.append({
        "time":                t,
        "type":                sub_type,
        "confidence":          confidence,
        "score":               score,
        "onset_strength":      onset,
        "energy":              energy,
        "bar_position_weight": bar_wt,
        "climax_bonus":        climax_bonus
    })

scored_points.sort(key=lambda x: x["score"], reverse=True)

print(f"    Total scored points: {len(scored_points)}")
print(f"    Top 5 accent points:")
for p in scored_points[:5]:
    print(f"      {p['time']}s — score {p['score']} "
          f"({p['type']}, {p['confidence']}, onset {p['onset_strength']}, "
          f"energy {p['energy']}, bar_wt {p['bar_position_weight']}, "
          f"climax_bonus {p['climax_bonus']})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Build prioritized accent map
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Building prioritized accent map...")

primary_candidates = [
    p for p in scored_points
    if is_downbeat(p["time"], downbeats)
]

# Secondary: non-downbeats, HIGH and MEDIUM confidence only (for effect placement)
# LOW confidence kept in JSON for analysis but excluded from effect candidates
secondary_candidates = [
    p for p in scored_points
    if not is_downbeat(p["time"], downbeats)
    and p["confidence"] in ("high", "medium")
]

secondary_low = [
    p for p in scored_points
    if not is_downbeat(p["time"], downbeats)
    and p["confidence"] == "low"
]

high_secondary   = sum(1 for p in secondary_candidates if p["confidence"] == "high")
medium_secondary = sum(1 for p in secondary_candidates if p["confidence"] == "medium")

print(f"    Primary   (downbeats only)        : {len(primary_candidates)}")
print(f"    Secondary HIGH+MEDIUM (for effects): {len(secondary_candidates)}")
print(f"      HIGH   (beat+peak coincide)     : {high_secondary}")
print(f"      MEDIUM (beat position only)     : {medium_secondary}")
print(f"    Secondary LOW (analysis only)     : {len(secondary_low)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Generate sync plans (2-8 photos)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Generating sync plans for 2-8 photos...")

def select_cut_points(candidates, n_cuts, duration, climax_time, min_gap=1.5):
    """
    Select n_cuts downbeat points following sync rules:
    - Always include climax downbeat
    - No cuts in bar 1 (already excluded from candidates)
    - For 2 photos: climax IS the only cut
    - Fill remaining by score with minimum gap
    - No cut in last 15% of track
    """
    selected   = []
    used_times = set()
    end_limit  = duration * 0.85

    valid = [p for p in candidates if p["time"] <= end_limit]

    # Rule 1: Always include climax point
    climax_candidates = sorted(valid, key=lambda x: abs(x["time"] - climax_time))
    if climax_candidates:
        best_climax = climax_candidates[0]
        selected.append(best_climax)
        used_times.add(best_climax["time"])

    if n_cuts == 1:
        return sorted(selected, key=lambda x: x["time"])

    # Rule 2: Add strong early point (first 35% of track)
    early_limit      = duration * 0.35
    early_candidates = [
        p for p in valid
        if p["time"] <= early_limit
        and p["time"] not in used_times
    ]
    if early_candidates:
        best_early = max(early_candidates, key=lambda x: x["score"])
        if not any(abs(best_early["time"] - s["time"]) < min_gap for s in selected):
            selected.append(best_early)
            used_times.add(best_early["time"])

    # Rule 3: Fill remaining by score with minimum gap
    remaining = [p for p in valid if p["time"] not in used_times]
    remaining.sort(key=lambda x: x["score"], reverse=True)

    for p in remaining:
        if len(selected) >= n_cuts:
            break
        too_close = any(abs(p["time"] - s["time"]) < min_gap for s in selected)
        if not too_close:
            selected.append(p)
            used_times.add(p["time"])

    return sorted(selected[:n_cuts], key=lambda x: x["time"])


sync_plans = {}
for n_photos in range(2, 9):
    n_cuts  = n_photos - 1
    min_gap = max(1.0, duration / (n_photos * 2))

    cut_points = select_cut_points(
        primary_candidates, n_cuts, duration, climax_time, min_gap
    )

    cut_times     = {p["time"] for p in cut_points}
    # Effect points: HIGH+MEDIUM secondary only, not overlapping cuts
    effect_points = [
        p for p in secondary_candidates
        if not any(abs(p["time"] - ct) < 0.1 for ct in cut_times)
    ][:20]

    sync_plans[str(n_photos)] = {
        "n_photos":      n_photos,
        "n_cuts":        len(cut_points),
        "cut_points":    [{"time": p["time"], "score": p["score"]}
                          for p in cut_points],
        "effect_points": [{"time": p["time"], "score": p["score"],
                           "type": p["type"], "confidence": p["confidence"]}
                          for p in effect_points]
    }

    cuts_str = [str(p["time"]) for p in cut_points]
    print(f"    {n_photos} photos → {len(cut_points)} cut(s) at: {', '.join(cuts_str)}s")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Save complete analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] Saving analysis...")

result = {
    "version":         "1.3",
    "file":            os.path.basename(TRACK),
    "duration":        round(duration, 3),
    "bpm":             bpm,
    "beat_duration":   beat_duration,
    "time_signature":  f"{time_signature}/4",
    "climax_time":     climax_time,
    "bar1_excluded":   True,
    "bar1_end":        bar1_end,
    "min_onset_threshold": MIN_ONSET_THRESHOLD,
    "attack_tolerance":    ATTACK_TOLERANCE,
    "track_range": {
        "energy_min":  round(rms_min, 4),
        "energy_max":  round(rms_max, 4),
        "energy_mean": round(rms_mean, 4),
        "onset_min":   round(onset_min, 4),
        "onset_max":   round(onset_max, 4)
    },
    "bars":       bars,
    "downbeats":  downbeats,
    "beat_times": beat_times,
    "rms_curve": {
        "times":  [round(float(t), 3) for t in rms_times.tolist()],
        "values": [round(v, 2) for v in rms_normalized]
    },
    "accent_map": {
        "primary":          primary_candidates[:30],
        "secondary":        secondary_candidates[:30],  # HIGH+MEDIUM combined (for visualizer)
        "secondary_high":   [p for p in secondary_candidates
                             if p["confidence"] == "high"][:30],
        "secondary_medium": [p for p in secondary_candidates
                             if p["confidence"] == "medium"][:30],
        "secondary_low":    secondary_low[:30]
    },
    "sync_plans": sync_plans
}

with open(OUTPUT, "w") as f:
    json.dump(result, f, indent=2)

print(f"    Saved: energetic_01_analysis.json")
print("\n" + "═"*50)
print("  ✓ Analysis complete v1.3")
print("═"*50 + "\n")
