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
VIDEO_FPS = 25
FONT_SIZE = 72
TEXT_COLOR = (255, 255, 255)
TEXT_OUTLINE_COLOR = (20, 20, 40)

# Background color varies by content pillar so posts read as visually distinct
# from each other, without needing fresh AI-generated art each time.
PILLAR_BACKGROUND_COLORS = {
    "Mock-philosophical wisdom": (58, 46, 82),    # deep moody indigo, sermon-like
    "Diva declarations": (196, 68, 122),          # glamorous magenta
    "Grumpy-thirsty one-liners": (214, 122, 40),  # punchy amber
    "Lore drops": (26, 28, 46),                   # near-black navy, mysterious
    "Boundary bits": (58, 150, 158),               # sharp icy teal
}

# Calibrated against reference/camp_yeti_reference.jpg specifically (2048x2732
# source) -- these are fractions of the ORIGINAL image's width/height, mapped
# through whatever scale render_text_card_image ends up using, so they track
# correctly regardless of final canvas size. Re-tune if the reference art changes.
BOW_ANCHOR_FRACTION = (0.53, 0.145)   # base of the head crest, where it meets the forehead
MOUTH_ANCHOR_FRACTION = (0.53, 0.26)  # between the fangs


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


def _draw_bow(draw: ImageDraw.ImageDraw, center: tuple, scale: float):
    """Draws a simple flat-vector pink bow -- matches the character's own flat
    cel-shaded illustration style rather than looking like a pasted-on sticker."""
    cx, cy = center
    w, h = 52 * scale, 34 * scale
    pink, outline = (232, 90, 156), (90, 30, 60)

    left_wing = [(cx, cy), (cx - w, cy - h / 2), (cx - w * 0.8, cy), (cx - w, cy + h / 2)]
    right_wing = [(cx, cy), (cx + w, cy - h / 2), (cx + w * 0.8, cy), (cx + w, cy + h / 2)]
    draw.polygon(left_wing, fill=pink, outline=outline, width=max(int(4 * scale), 2))
    draw.polygon(right_wing, fill=pink, outline=outline, width=max(int(4 * scale), 2))

    knot_w, knot_h = 22 * scale, 26 * scale
    draw.ellipse(
        [cx - knot_w / 2, cy - knot_h / 2, cx + knot_w / 2, cy + knot_h / 2],
        fill=pink, outline=outline, width=max(int(4 * scale), 2),
    )


def _draw_rose(draw: ImageDraw.ImageDraw, mouth_point: tuple, scale: float):
    """Draws a small stylized rose hanging from the fangs -- sinister-glamour
    detail rather than gore, flat-shaded to match the illustration style."""
    mx, my = mouth_point
    stem_bottom = (mx + 4 * scale, my + 90 * scale)
    draw.line([mouth_point, stem_bottom], fill=(60, 110, 60), width=max(int(6 * scale), 3))

    leaf_w, leaf_h = 20 * scale, 10 * scale
    leaf_y = my + 55 * scale
    draw.ellipse(
        [mx - leaf_w, leaf_y - leaf_h / 2, mx, leaf_y + leaf_h / 2],
        fill=(70, 130, 70), outline=(40, 80, 40),
    )

    petal_color, petal_outline = (176, 30, 55), (90, 10, 30)
    r = 16 * scale
    petal_offsets = [(0, 0), (-r * 0.7, -r * 0.3), (r * 0.7, -r * 0.3), (0, -r * 0.6)]
    for ox, oy in petal_offsets:
        px, py = mx + ox, my + oy
        draw.ellipse([px - r, py - r, px + r, py + r], fill=petal_color, outline=petal_outline, width=2)


def _cutout_character(source: Image.Image, tolerance: int = 40) -> Image.Image:
    """Removes the reference photo's own flat backdrop via flood-fill from
    each corner -- only removes background pixels actually connected to the
    border, so similarly-colored fur (e.g. the cream chest) inside the
    silhouette is left alone, unlike a global color-distance match."""
    rgba = source.convert("RGBA")
    corners = [(0, 0), (rgba.width - 1, 0), (0, rgba.height - 1), (rgba.width - 1, rgba.height - 1)]
    for corner in corners:
        ImageDraw.floodfill(rgba, corner, (0, 0, 0, 0), thresh=tolerance)
    return rgba


def render_text_card_image(image_text: str, pillar: str = None) -> Path:
    """Overlays image_text onto the reference character art -- entirely local,
    no AI image generation, so the art is always the same approved portrait.

    Cuts the character out of its own flat backdrop and places it on a
    pillar-themed background color, then fits (not crops) it into the lower
    portion of the frame, guaranteeing clear headroom above for the text
    regardless of how many lines it wraps to.
    """
    source = Image.open(REFERENCE_IMAGE_PATH).convert("RGB")
    source_bg_color = source.getpixel((5, 5))
    cutout = _cutout_character(source)

    canvas_color = PILLAR_BACKGROUND_COLORS.get(pillar, source_bg_color)
    canvas = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), canvas_color)
    top_margin = 360  # reserved for text; character fits below this

    available_h = VIDEO_HEIGHT - top_margin
    scale = min(VIDEO_WIDTH / cutout.width, available_h / cutout.height)
    resized = cutout.resize((int(cutout.width * scale), int(cutout.height * scale)), Image.LANCZOS)
    paste_x = (VIDEO_WIDTH - resized.width) // 2
    paste_y = VIDEO_HEIGHT - resized.height  # anchor to the bottom
    canvas.paste(resized, (paste_x, paste_y), mask=resized)

    draw = ImageDraw.Draw(canvas)

    bow_center = (
        paste_x + BOW_ANCHOR_FRACTION[0] * resized.width,
        paste_y + BOW_ANCHOR_FRACTION[1] * resized.height,
    )
    _draw_bow(draw, bow_center, scale=resized.width / 742)

    mouth_point = (
        paste_x + MOUTH_ANCHOR_FRACTION[0] * resized.width,
        paste_y + MOUTH_ANCHOR_FRACTION[1] * resized.height,
    )
    _draw_rose(draw, mouth_point, scale=resized.width / 742)

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
    frame_count = VIDEO_DURATION_SECONDS * VIDEO_FPS
    # Slow, subtle zoom-in (Ken Burns style) so the still image reads as a
    # video rather than a static photo -- scale up first so zoompan doesn't
    # introduce its own upscale artifacts.
    zoompan = (
        f"scale={VIDEO_WIDTH * 2}:{VIDEO_HEIGHT * 2},"
        f"zoompan=z='min(zoom+0.0006,1.08)':d={frame_count}:"
        f"s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}"
    )
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(image_path),
                "-i", str(music_path),
                "-t", str(VIDEO_DURATION_SECONDS),
                "-vf", zoompan,
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
        image_path = render_text_card_image(post["image_text"], pillar=post.get("pillar_used"))
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
