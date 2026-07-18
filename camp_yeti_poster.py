"""
Camp Yeti autonomous poster.
Reads the persona bible, asks Claude for a caption + a short text-card line,
overlays that text onto the fixed reference character art (no AI image
generation -- same approved artwork every time), sets it to a music track,
and publishes the resulting video to Instagram via Blotato -- used only for
hosting + publishing, never for AI generation, since that's what costs money.

Env vars required:
  ANTHROPIC_API_KEY
  BLOTATO_API_KEY
  BLOTATO_INSTAGRAM_ACCOUNT_ID   (the accountId from Blotato's Accounts page)

Also requires ffmpeg on PATH, and `pip install Pillow`.

Run this on a schedule (see .github/workflows/camp_yeti_post.yml).
"""

import os
import json
import time
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
BLOTATO_API_KEY = os.environ["BLOTATO_API_KEY"]
BLOTATO_ACCOUNT_ID = os.environ["BLOTATO_INSTAGRAM_ACCOUNT_ID"]

PERSONA_PATH = Path(__file__).parent / "camp_yeti_persona_bible.md"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "camp_yeti_agent_system_prompt.md"
REFERENCE_IMAGE_PATH = Path(__file__).parent / "reference" / "camp_yeti_reference.jpg"
FONT_PATH = Path(__file__).parent / "fonts" / "Anton-Regular.ttf"
MUSIC_DIR = Path(__file__).parent / "music"

ANTHROPIC_MODEL = "claude-sonnet-4-6"

VIDEO_WIDTH, VIDEO_HEIGHT = 1080, 1350  # 4:5 -- works for both feed and reel
VIDEO_DURATION_SECONDS = 15
FONT_SIZE = 72
TEXT_COLOR = (255, 255, 255)
TEXT_OUTLINE_COLOR = (20, 20, 40)


def _raise_with_body(resp: requests.Response):
    """resp.raise_for_status() only reports the status code -- APIs put the
    actually-useful detail (which field was invalid, why) in the response
    body, so surface that in the exception instead of losing it."""
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{e} -- {resp.text}", response=resp) from e


def call_claude(system_prompt: str, user_prompt: str) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=60,
    )
    _raise_with_body(resp)
    data = resp.json()
    return "".join(b["text"] for b in data["content"] if b["type"] == "text")


def generate_post(persona_bible: str) -> dict:
    """Ask Claude for a JSON-structured post: caption + short on-image text-card line."""
    system = (
        "You are the autonomous content generator for the Camp Yeti Instagram "
        "persona. Follow the persona bible exactly. Respond with ONLY valid JSON, "
        "no markdown fences, no preamble, matching this schema:\n"
        '{"caption": "...", "image_text": "...", "pillar_used": "...", '
        '"phrase_used": "... or null", "new_lore": "... or null"}'
    )
    user = (
        f"PERSONA BIBLE:\n{persona_bible}\n\n"
        "Generate today's post. Pick a pillar not used in the last 3 log entries. "
        "This is a text-card style post over a fixed portrait of Yeti -- no new "
        "artwork is generated, so image_text carries the whole joke.\n\n"
        "image_text: the short, punchy line(s) that appear ON the image itself "
        "(like the reference posts -- 'WINTER COLLECTION. SPRING COLLECTION... "
        "DARLING, I am the collection.'). 1-4 short lines. This is what she's "
        "declaring, in her voice per the Voice Rules section.\n\n"
        "caption is the separate Instagram post caption (different text, not "
        "just a repeat of image_text) -- 1-4 lines, #CAMPYETI plus at most one "
        "or two theme tags."
    )
    raw = call_claude(system, user)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    words = text.split()
    lines = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_text_card_image(image_text: str) -> Path:
    """Overlays image_text onto the reference character art -- entirely local,
    no AI image generation, so the art is always the same approved portrait.

    Fits (not crops) the character into the lower portion of the frame on a
    canvas extending the image's own background color, guaranteeing clear
    headroom above for the text regardless of how many lines it wraps to.
    """
    source = Image.open(REFERENCE_IMAGE_PATH).convert("RGB")
    background_color = source.getpixel((5, 5))

    canvas = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), background_color)
    top_margin = 360  # reserved for text; character fits below this

    available_h = VIDEO_HEIGHT - top_margin
    scale = min(VIDEO_WIDTH / source.width, available_h / source.height)
    resized = source.resize((int(source.width * scale), int(source.height * scale)), Image.LANCZOS)
    paste_x = (VIDEO_WIDTH - resized.width) // 2
    paste_y = VIDEO_HEIGHT - resized.height  # anchor to the bottom
    canvas.paste(resized, (paste_x, paste_y))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)

    lines = []
    for raw_line in image_text.split("\n"):
        lines.extend(_wrap_text(draw, raw_line.upper(), font, VIDEO_WIDTH - 120))

    line_height = FONT_SIZE + 16
    total_text_height = line_height * len(lines)
    y = max((top_margin - total_text_height) // 2, 40)
    for line in lines:
        width = draw.textlength(line, font=font)
        x = (VIDEO_WIDTH - width) / 2
        for dx, dy in [(-3, -3), (-3, 3), (3, -3), (3, 3), (-3, 0), (3, 0), (0, -3), (0, 3)]:
            draw.text((x + dx, y + dy), line, font=font, fill=TEXT_OUTLINE_COLOR)
        draw.text((x, y), line, font=font, fill=TEXT_COLOR)
        y += line_height

    out_path = Path(tempfile.mkdtemp(prefix="camp-yeti-")) / "text_card.jpg"
    canvas.save(out_path, quality=95)
    return out_path


def build_video(image_path: Path) -> Path:
    """Loops the text-card image for VIDEO_DURATION_SECONDS over a random
    track from music/ -- entirely local via ffmpeg, no Blotato credits."""
    tracks = sorted(MUSIC_DIR.glob("*.mp3"))
    if not tracks:
        raise RuntimeError(
            f"No music tracks found in {MUSIC_DIR} -- add at least one .mp3 "
            "(see music/README.md)."
        )
    music_path = random.choice(tracks)

    out_path = image_path.parent / "final.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(image_path),
                "-i", str(music_path),
                "-t", str(VIDEO_DURATION_SECONDS),
                "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                str(out_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed building video: {e.stderr}")

    return out_path


def upload_video_to_blotato(video_path: Path) -> str:
    """Uploads a local video to Blotato via its presigned-upload endpoint --
    free, since only AI generation is billed, not hosting/publishing."""
    upload_start = requests.post(
        "https://backend.blotato.com/v2/media/uploads",
        headers={
            "blotato-api-key": BLOTATO_API_KEY,
            "Content-Type": "application/json",
        },
        json={"filename": "camp-yeti-post.mp4"},
        timeout=30,
    )
    _raise_with_body(upload_start)
    upload_info = upload_start.json()

    with open(video_path, "rb") as f:
        put_resp = requests.put(
            upload_info["presignedUrl"],
            data=f,
            headers={"Content-Type": "video/mp4"},
            timeout=120,
        )
    _raise_with_body(put_resp)

    return upload_info["publicUrl"]


def publish_to_instagram(caption: str, video_url: str) -> dict:
    resp = requests.post(
        "https://backend.blotato.com/v2/posts",
        headers={
            "blotato-api-key": BLOTATO_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "post": {
                "accountId": BLOTATO_ACCOUNT_ID,
                "content": {
                    "text": caption,
                    "mediaUrls": [video_url],
                    "platform": "instagram",
                },
                "target": {"targetType": "instagram", "mediaType": "reel"},
            }
        },
        timeout=30,
    )
    _raise_with_body(resp)
    return resp.json()


def append_log(entry: dict):
    text = PERSONA_PATH.read_text()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    line = (
        f"- {date} | pillar: {entry['pillar_used']} | "
        f"phrase: {entry.get('phrase_used') or 'none'} | "
        f"new lore: {entry.get('new_lore') or 'none'}"
    )
    text = text.replace("- [log starts empty]", line + "\n- [log starts empty]")
    PERSONA_PATH.write_text(text)


def notify_owner(message: str):
    """Wire this to email/Slack/SMS -- whatever reaches you. Placeholder: prints."""
    print(f"[ESCALATION] {message}")
    # e.g. requests.post(SLACK_WEBHOOK_URL, json={"text": message})


def main():
    persona_bible = PERSONA_PATH.read_text()

    for attempt in range(3):
        try:
            post = generate_post(persona_bible)
            break
        except Exception as e:
            if attempt == 2:
                notify_owner(f"Post generation failed 3x: {e}")
                return
            time.sleep(2)

    try:
        image_path = render_text_card_image(post["image_text"])
        video_path = build_video(image_path)
    except Exception as e:
        notify_owner(f"Video assembly failed: {e}")
        return

    try:
        video_url = upload_video_to_blotato(video_path)
    except Exception as e:
        notify_owner(f"Video upload failed: {e}")
        return

    try:
        result = publish_to_instagram(post["caption"], video_url)
    except requests.HTTPError as e:
        notify_owner(f"Blotato publish failed: {e}")
        return

    append_log(post)
    print(f"Published: {result}")


if __name__ == "__main__":
    main()
