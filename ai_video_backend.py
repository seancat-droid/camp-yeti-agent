"""
PROTOTYPE -- real AI-generated video as a candidate replacement for the
local bounce/blink/zoom animation in build_animated_video()
(camp_yeti_poster.py).

STATUS: not wired in as the live default. camp_yeti_poster.py only reaches
this module when CAMP_YETI_AI_VIDEO_PROVIDER is set in the environment --
it isn't set anywhere in this repo (not in the Actions workflow, not in any
committed config), so the live posting pipeline is completely unaffected by
this file existing. Turning it on costs real per-video money and requires a
paid API account; this code does not sign up for one or spend anything on
its own. See IMPROVEMENT_LOG.md for the full cost writeup and exactly what
a human needs to do to activate it.

Why this instead of the current pipeline: the hand-animated version is real
video (ffmpeg + beat-synced motion), but it's always the same flat character
cutout bouncing in place -- no camera-aware parallax, no secondary motion
(fur/fabric sway, hand gestures), no scene depth. An image-to-video model
given the exact composited frame (character + text + background, as
produced by _build_scene in camp_yeti_poster.py) can add that motion while
keeping the *same* reference artwork, so the character design the account
is built around doesn't change.

Researched options (Nov 2026 -- sourced from third-party pricing blogs that
disagree with each other; confirm against each vendor's own pricing page
before paying anything):

  * Runway Gen-4 Turbo (image-to-video)   ~$0.05/sec  -- implemented below
  * Kling 3.0 / O1                        ~$0.08-0.17/sec -- vendor claims
    strong character consistency, but priced noticeably higher
  * Google Veo 3.1 Lite                   ~$0.03/sec (720p, video-only)
  * MiniMax Hailuo 02                     ~$0.017-0.045/sec -- cheapest found,
    but no independent read on character-consistency quality
  * ByteDance Seedance 2.0 Fast           ~$0.022-0.09/sec depending on route

Runway Gen-4 Turbo is prototyped here because it has the most stable,
well-documented API among this list and a plain image+text-prompt -> video
call. Swapping providers later means swapping _call_runway() below, not
anything in camp_yeti_poster.py -- build_ai_video_clip() is the only
function the caller depends on.

Known limitation: Gen-4 Turbo caps a single generation at 10 seconds, so
this prototype produces one clip of up to 10s, not the account's usual
16-30s reel. Chaining multiple generated clips into a longer reel is future
work, left for whoever picks this up next if the owner decides to activate
it.

IMPORTANT: the exact Runway request/response field names below were
researched, not verified against a live call (this environment couldn't
reach docs.dev.runwayml.com to confirm). Re-check field names against
Runway's current API docs before relying on this in anything real.
"""

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests

RUNWAY_API_KEY = os.environ.get("RUNWAY_API_KEY")  # intentionally unset by default
RUNWAY_API_BASE = "https://api.dev.runwayml.com/v1"
RUNWAY_API_VERSION = "2024-11-06"  # verify against current docs before activating

# gen4_turbo only accepts fixed aspect ratios -- none is an exact match for
# this account's 1080x1350 (4:5) feed/reel canvas, so we generate at the
# closest supported portrait ratio and letterbox/pad back to 4:5 afterward
# (see build_ai_video_clip's ffmpeg mux step).
RUNWAY_RATIO = "832:1104"


def _image_to_data_uri(image_path: Path) -> str:
    suffix = image_path.suffix.lstrip(".").lower() or "jpeg"
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _call_runway(image_path: Path, motion_prompt: str, duration_seconds: int,
                  poll_interval: int = 5, timeout: int = 300) -> str:
    """Submits a Gen-4 Turbo image-to-video job and polls until it finishes.
    Returns the generated video's download URL. Raises on any failure --
    callers are expected to fall back to the local animation pipeline."""
    if not RUNWAY_API_KEY:
        raise RuntimeError(
            "RUNWAY_API_KEY is not set -- AI video generation is not configured. "
            "See IMPROVEMENT_LOG.md for what activating this would involve."
        )
    duration = 10 if duration_seconds > 7 else 5  # gen4_turbo only accepts 5 or 10
    headers = {
        "Authorization": f"Bearer {RUNWAY_API_KEY}",
        "X-Runway-Version": RUNWAY_API_VERSION,
        "Content-Type": "application/json",
    }
    body = {
        "model": "gen4_turbo",
        "promptImage": _image_to_data_uri(image_path),
        "promptText": motion_prompt,
        "ratio": RUNWAY_RATIO,
        "duration": duration,
    }
    resp = requests.post(f"{RUNWAY_API_BASE}/image_to_video", headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    task_id = resp.json()["id"]

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        poll = requests.get(f"{RUNWAY_API_BASE}/tasks/{task_id}", headers=headers, timeout=30)
        poll.raise_for_status()
        data = poll.json()
        status = data.get("status")
        if status == "SUCCEEDED":
            return data["output"][0]
        if status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Runway generation {status.lower()}: {data.get('failure', data)}")
    raise TimeoutError(f"Runway generation for task {task_id} did not finish within {timeout}s")


def build_ai_video_clip(still_frame_path: Path, motion_prompt: str, music_path: Path,
                         duration_seconds: int, video_width: int, video_height: int) -> Path:
    """End-to-end: AI-animates still_frame_path, downloads the (silent)
    result, pads it to (video_width, video_height), and muxes in music_path
    as audio. Returns the final mp4 path. Raises on any failure (missing
    key, API error, network issue, ffmpeg failure) -- the caller in
    camp_yeti_poster.py catches this and falls back to the local
    hand-animated pipeline, so a broken or unconfigured AI backend never
    blocks a post."""
    video_url = _call_runway(still_frame_path, motion_prompt, duration_seconds)
    raw_bytes = requests.get(video_url, timeout=60).content

    work_dir = Path(tempfile.mkdtemp(prefix="camp-yeti-ai-"))
    raw_path = work_dir / "ai_raw.mp4"
    raw_path.write_bytes(raw_bytes)

    out_path = work_dir / "ai_final.mp4"
    pad_filter = (
        f"scale={video_width}:{video_height}:force_original_aspect_ratio=decrease,"
        f"pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(raw_path),
            "-i", str(music_path),
            "-filter:v", pad_filter,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(out_path),
        ],
        check=True, capture_output=True,
    )
    return out_path
