import subprocess
import json
import os
import glob

# ── CONFIG ────────────────────────────────────────
BASE_DIR     = os.path.expanduser("~/Documents/advids")
TRACKS_DIR   = os.path.join(BASE_DIR, "tracks")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
PHOTOS_DIR   = os.path.join(BASE_DIR, "photos")
TESTS_DIR    = os.path.join(BASE_DIR, "tests")
TEMP_DIR     = os.path.join(BASE_DIR, "temp")

ANALYSIS   = os.path.join(ANALYSIS_DIR, "energetic_01_analysis.json")
TRACK      = os.path.join(TRACKS_DIR,   "energetic_01.mp3")
OUTPUT_DIR = os.path.join(TESTS_DIR,    "output")
WIDTH, HEIGHT = 1080, 1920  # 9:16 vertical

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ── LOAD ANALYSIS ─────────────────────────────────
with open(ANALYSIS) as f:
    analysis = json.load(f)

duration = analysis["duration"]
section_start = 0  # use full track for now

# ── FIND PHOTOS ───────────────────────────────────
extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
photos = []
for ext in extensions:
        photos.extend(glob.glob(os.path.join(PHOTOS_DIR, ext)))
photos = sorted(photos)

if len(photos) < 2:
      print("❌ Need at least 2 photos in ~/Documents/advids/photos/")
    exit()

print(f"\n{'═'*50}")
print(f"  AdVids — Sync Test")
print(f"{'═'*50}")
print(f"  Photos found: {len(photos)}")
for p in photos:
    print(f"    → {os.path.basename(p)}")

# ── TEST ALL PHOTO COUNTS ─────────────────────────
# Test with however many photos we have (up to 8)
test_counts = list(range(2, min(len(photos) + 1, 9)))

for n_photos in test_counts:
    print(f"\n{'─'*50}")
    print(f"  Testing: {n_photos} photos")

    # Get sync plan for this photo count
    plan = analysis["sync_plans"][str(n_photos)]
    cut_points = [p["time"] for p in plan["cut_points"]]

    print(f"  Cut points: {cut_points}")

    # Select photos for this test
    selected_photos = photos[:n_photos]

    # Build timeline: list of (photo_path, start_time, end_time)
    timeline = []
    starts = [0.0] + cut_points
    ends   = cut_points + [duration]

    for i, photo in enumerate(selected_photos):
        timeline.append({
            "photo": photo,
            "start": starts[i],
            "end":   ends[i],
            "duration": round(ends[i] - starts[i], 3)
        })
        print(f"    Photo {i+1}: {os.path.basename(photo)} "
              f"[{round(starts[i],2)}s → {round(ends[i],2)}s] "
              f"({round(ends[i]-starts[i],2)}s)")

    # ── BUILD CLIPS ───────────────────────────────
    clip_paths = []
    for i, item in enumerate(timeline):
        clip_path = os.path.join(TEMP_DIR, f"clip_{n_photos}p_{i:02d}.mp4")

        # Simple static photo — no effects, pure sync test
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", item["photo"],
            "-vf", (
                f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{HEIGHT}"
            ),
            "-t", str(item["duration"]),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",  # fast for testing
            clip_path
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            print(f"    ❌ Error building clip {i}: {result.stderr.decode()[-200:]}")
            continue

        clip_paths.append(clip_path)
        print(f"    ✓ Clip {i+1} built ({item['duration']}s)")

    if len(clip_paths) != n_photos:
        print(f"  ❌ Not all clips built. Skipping {n_photos} photo test.")
        continue

    # ── CONCATENATE CLIPS ─────────────────────────
    concat_list = os.path.join(TEMP_DIR, f"concat_{n_photos}p.txt")
    with open(concat_list, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    concat_output = os.path.join(TEMP_DIR, f"concat_{n_photos}p.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        concat_output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── ADD AUDIO ─────────────────────────────────
    final_output = os.path.join(OUTPUT_DIR, f"test_{n_photos}photos.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", concat_output,
        "-i", TRACK,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        final_output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"  ✓ Output: test_{n_photos}photos.mp4")

print(f"\n{'═'*50}")
print(f"  ✓ All tests complete")
print(f"  Output folder: ~/Documents/advids/tests/output/")
print(f"{'═'*50}\n")