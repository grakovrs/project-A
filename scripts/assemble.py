#!/usr/bin/env python3
"""
assemble.py — AdVids Video Assembler v1.0
Takes N photos + analysis JSON + mp3 → outputs mp4 with beat-synced hard cuts.

Usage:
  python assemble.py photo1.jpg photo2.jpg ... --json analysis.json --audio track.mp3 --output out.mp4

The number of photos N determines which sync_plan is used from the JSON.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def load_analysis(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_cut_points(analysis, n_photos):
    key = str(n_photos)
    plans = analysis.get("sync_plans", {})
    if key not in plans:
        available = list(plans.keys())
        print(f"ERROR: No sync plan for {n_photos} photos. Available: {available}")
        sys.exit(1)
    plan = plans[key]
    cut_points = plan["cut_points"]
    print(f"Sync plan for {n_photos} photos: {len(cut_points)} cut points")
    for i, cp in enumerate(cut_points):
        print(f"  Cut {i+1}: {cp['time']:.3f}s  score={cp.get('score', 0):.1f}")
    return cut_points


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def build_video(photos, cut_points, audio_path, output_path):
    """
    Build mp4 using FFmpeg concat demuxer.
    Each photo is shown from one cut point to the next.
    Photo 1 starts at time 0, switches at cut_points[0].
    Photo 2 shown from cut_points[0] to cut_points[1], etc.
    Last photo shown from last cut point to end of audio.
    """
    n = len(photos)
    audio_duration = get_audio_duration(audio_path)
    print(f"\nAudio duration: {audio_duration:.3f}s")

    # Build segment start times
    # Photo 1: 0 → cut_points[0]
    # Photo i: cut_points[i-1] → cut_points[i]
    # Photo n: cut_points[n-2] → end
    starts = [0.0] + [cp["time"] for cp in cut_points]
    ends = [cp["time"] for cp in cut_points] + [audio_duration]

    if len(starts) != n:
        print(f"ERROR: {n} photos but need {n} segments. Got {len(starts)} starts.")
        print(f"  cut_points count: {len(cut_points)}, need exactly {n-1}")
        sys.exit(1)

    durations = [ends[i] - starts[i] for i in range(n)]
    print(f"\nSegments:")
    for i in range(n):
        print(f"  Photo {i+1}: {starts[i]:.3f}s → {ends[i]:.3f}s  ({durations[i]:.3f}s)  [{os.path.basename(photos[i])}]")

    # Write concat list file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = f.name
        for i in range(n):
            abs_photo = os.path.abspath(photos[i])
            f.write(f"file '{abs_photo}'\n")
            f.write(f"duration {durations[i]:.6f}\n")
        # FFmpeg concat demuxer requires last file listed twice (no duration on last)
        abs_photo = os.path.abspath(photos[-1])
        f.write(f"file '{abs_photo}'\n")

    print(f"\nBuilding video...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_file,
        "-i", audio_path,
        "-c:v", "libx264", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path
    ]
    print(f"FFmpeg command:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    os.unlink(concat_file)

    if result.returncode != 0:
        print(f"\nERROR: FFmpeg failed with code {result.returncode}")
        sys.exit(1)

    print(f"\nDone! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AdVids Assembler v1.0")
    parser.add_argument("photos", nargs="+", help="Photo files in order (1 to 8)")
    parser.add_argument("--json", required=True, help="Analysis JSON from analyze2.py")
    parser.add_argument("--audio", required=True, help="Audio mp3 file")
    parser.add_argument("--output", default="output.mp4", help="Output mp4 path")
    args = parser.parse_args()

    n = len(args.photos)
    if n < 1 or n > 8:
        print(f"ERROR: Need 1–8 photos, got {n}")
        sys.exit(1)

    # Verify files exist
    for p in args.photos:
        if not os.path.exists(p):
            print(f"ERROR: Photo not found: {p}")
            sys.exit(1)
    if not os.path.exists(args.json):
        print(f"ERROR: JSON not found: {args.json}")
        sys.exit(1)
    if not os.path.exists(args.audio):
        print(f"ERROR: Audio not found: {args.audio}")
        sys.exit(1)

    print(f"AdVids Assembler v1.0")
    print(f"Photos ({n}): {[os.path.basename(p) for p in args.photos]}")
    print(f"JSON:  {args.json}")
    print(f"Audio: {args.audio}")
    print(f"Output: {args.output}")
    print()

    analysis = load_analysis(args.json)
    cut_points = get_cut_points(analysis, n)

    # n photos need n-1 cut points
    if len(cut_points) < n - 1:
        print(f"WARNING: Only {len(cut_points)} cut points for {n} photos (need {n-1}). Using first {n-1}.")
        cut_points = cut_points[:n-1]
    elif len(cut_points) > n - 1:
        print(f"NOTE: {len(cut_points)} cut points available, using first {n-1} for {n} photos.")
        cut_points = cut_points[:n-1]

    build_video(args.photos, cut_points, args.audio, args.output)


if __name__ == "__main__":
    main()
