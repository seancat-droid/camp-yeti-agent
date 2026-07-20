"""
Camp Yeti autonomous poster.
Reads the persona bible, asks Claude for a caption + a short text-card line,
overlays that text onto the fixed reference character art (no AI image
generation -- same approved artwork every time), sets it to a music track,
and publishes the resulting video to Instagram, TikTok, YouTube, and Facebook
concurrently via Blotato -- used only for hosting + publishing, never for AI
generation, since that's what costs money.

Env vars required:
  ANTHROPIC_API_KEY
  BLOTATO_API_KEY
  BLOTATO_INSTAGRAM_ACCOUNT_ID   (the accountId from Blotato's Accounts page)

Optional (default to the accounts connected when this was built -- override
if you reconnect any of them):
  BLOTATO_TIKTOK_ACCOUNT_ID
  BLOTATO_YOUTUBE_ACCOUNT_ID
  BLOTATO_FACEBOOK_ACCOUNT_ID
  BLOTATO_FACEBOOK_PAGE_ID

Also requires ffmpeg on PATH, and `pip install Pillow`.

Run this on a schedule (see .github/workflows/camp_yeti_post.yml).
"""

import os
import json
import time
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
BLOTATO_API_KEY = os.environ["BLOTATO_API_KEY"]

# Account IDs per platform, from GET /v2/users/me/accounts. Defaults are the
# accounts connected as of this writing; override via env var if reconnected.
BLOTATO_ACCOUNT_IDS = {
    "instagram": os.environ["BLOTATO_INSTAGRAM_ACCOUNT_ID"],
    "tiktok": os.environ.get("BLOTATO_TIKTOK_ACCOUNT_ID", "51690"),
    "youtube": os.environ.get("BLOTATO_YOUTUBE_ACCOUNT_ID", "43870"),
    "facebook": os.environ.get("BLOTATO_FACEBOOK_ACCOUNT_ID", "41929"),
}
BLOTATO_FACEBOOK_PAGE_ID = os.environ.get("BLOTATO_FACEBOOK_PAGE_ID", "41929")

PERSONA_PATH = Path(__file__).parent / "camp_yeti_persona_bible.md"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "camp_yeti_agent_system_prompt.md"
REFERENCE_DIR = Path(__file__).parent / "reference"
FONT_PATH = Path(__file__).parent / "fonts" / "Anton-Regular.ttf"
MUSIC_DIR = Path(__file__).parent / "music"

ANTHROPIC_MODEL = "claude-sonnet-4-6"

VIDEO_WIDTH, VIDEO_HEIGHT = 1080, 1350  # 4:5 -- works for both feed and reel
MAX_VIDEO_DURATION_SECONDS = 90  # Instagram's Reels eligibility cap
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

# Bow colorway rotates per post -- (fill, outline) -- so the one recurring
# prop doesn't make every post look like a carbon copy of the last.
BOW_COLORWAYS = [
    ((232, 90, 156), (90, 30, 60)),   # classic pink
    ((200, 40, 70), (70, 10, 25)),    # ruby red
    ((218, 165, 32), (90, 60, 10)),   # gold
    ((58, 150, 158), (15, 60, 65)),   # teal
    ((130, 70, 170), (50, 20, 70)),   # violet
]

# Ken Burns motion presets for build_video -- rotates so posts don't all pan
# the exact same way.
ZOOMPAN_MOTION_PRESETS = ["zoom_in", "zoom_in_pan_right", "zoom_in_pan_left", "zoom_out"]

# Looks to rotate between. bow_anchor/eyes_anchor/necklace_anchor/hand_anchor
# are fractions of each image's own width/height (calibrated per-image since
# proportions differ), mapped through whatever scale render_text_card_image
# ends up using, so they track correctly regardless of final canvas size.
#
# The photorealistic look was retired -- its chest render read as
# sexualized/off-model and Blotato's image template schema isn't reliably
# reverse-engineerable, so it's not worth chasing a fix.
REFERENCE_LOOKS = [
    {
        "path": REFERENCE_DIR / "camp_yeti_reference.jpg",
        "bow_anchor": (0.53, 0.145),  # base of the head crest, where it meets the forehead
        "eyes_anchor": (0.439, 0.203),  # midpoint between the eyes, for sunglasses
        "eyes_span": 0.107,  # fraction of width between the two eyes, for sizing
        "necklace_anchor": (0.415, 0.260),  # base of the neck, for a pearl string
        "necklace_span": 0.19,
        "hand_anchor": (0.19, 0.77),  # left fist, for a handbag
        "weight": 1,
    },
]

# One rotating "hero prop" per the persona bible's accessory rule (never more
# than one at once, on top of the always-on bow). Weighted toward "none" so
# it stays occasional rather than cluttering the silhouette every time.
ACCESSORY_POOL = ["none", "none", "sunglasses", "necklace", "handbag"]
PEARL_COLORWAY = ((238, 230, 210), (150, 130, 90))

# Background styles rotate independently of pillar color so posts don't all
# read as "flat color card" every time.
BACKGROUND_STYLES = ["solid", "solid", "vertical_gradient", "radial_spotlight", "diagonal_split", "glitter"]


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



# Structural formats to rotate between, independent of pillar (topic). Without
# this, generation converges on the same rule-of-three mock-sermon shape every
# time even when the topic changes -- format variety is what actually breaks
# the "these all sound the same" pattern.
CAPTION_FORMATS = [
    "single blunt line -- one devastating line, no build-up, no undercut needed because the line already lands flat",
    "rule-of-three build -- the classic mock-sermon that gets undercut by one blunt gag line at the end",
    "direct address -- speaks straight to the reader in second person, like she's calling someone out specifically",
    "mock testimonial -- framed as if quoting or reporting what someone else said/did in reaction to her",
    "implied list -- a short run of parallel grievances or rules, stated like an itemized list without literal numbering",
    "confession -- opens by admitting or conceding something about herself before the real joke lands",
]


def generate_post(persona_bible: str) -> dict:
    """Ask Claude for a JSON-structured post: caption + short on-image text-card line."""
    formats_block = "\n".join(f"- {f}" for f in CAPTION_FORMATS)
    system = (
        "You are the autonomous content generator for the Camp Yeti Instagram "
        "persona. Follow the persona bible exactly. Respond with ONLY valid JSON, "
        "no markdown fences, no preamble, matching this schema:\n"
        '{"caption": "...", "image_text": "...", "pillar_used": "...", '
        '"format_used": "...", "phrase_used": "... or null", "new_lore": "... or null"}'
    )
    user = (
        f"PERSONA BIBLE:\n{persona_bible}\n\n"
        "Generate today's post. Pick a pillar not used in the last 3 log entries. "
        "Separately, pick a structural FORMAT not used in the last 3 log entries "
        f"(check the log's 'format:' field) from:\n{formats_block}\n\n"
        "Vary length too -- some posts should be one short line, others should "
        "build across several. Don't default to the longest, most elaborate "
        "option every time. This is a text-card style post over a fixed "
        "portrait of Yeti -- no new artwork is generated, so image_text "
        "carries the whole joke.\n\n"
        "image_text: the short, punchy line(s) that appear ON the image itself "
        "(like the reference posts -- 'WINTER COLLECTION. SPRING COLLECTION... "
        "DARLING, I am the collection.'). 1-4 short lines. This is what she's "
        "declaring, in her voice per the Voice Rules section, in the format "
        "you picked.\n\n"
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


def _draw_bow(draw: ImageDraw.ImageDraw, center: tuple, scale: float, colorway: tuple = None):
    """Draws a simple flat-vector bow -- matches the character's own flat
    cel-shaded illustration style rather than looking like a pasted-on sticker."""
    cx, cy = center
    w, h = 52 * scale, 34 * scale
    pink, outline = colorway or BOW_COLORWAYS[0]

    left_wing = [(cx, cy), (cx - w, cy - h / 2), (cx - w * 0.8, cy), (cx - w, cy + h / 2)]
    right_wing = [(cx, cy), (cx + w, cy - h / 2), (cx + w * 0.8, cy), (cx + w, cy + h / 2)]
    draw.polygon(left_wing, fill=pink, outline=outline, width=max(int(4 * scale), 2))
    draw.polygon(right_wing, fill=pink, outline=outline, width=max(int(4 * scale), 2))

    knot_w, knot_h = 22 * scale, 26 * scale
    draw.ellipse(
        [cx - knot_w / 2, cy - knot_h / 2, cx + knot_w / 2, cy + knot_h / 2],
        fill=pink, outline=outline, width=max(int(4 * scale), 2),
    )


def _draw_heart(draw: ImageDraw.ImageDraw, center: tuple, size: float, colorway: tuple):
    """Flat vector heart -- two lobes plus a triangular point, used as each
    sunglasses lens."""
    cx, cy = center
    fill, outline = colorway
    r = size * 0.5
    width = max(int(size * 0.08), 2)
    draw.ellipse([cx - r, cy - r * 0.7, cx, cy + r * 0.3], fill=fill, outline=outline, width=width)
    draw.ellipse([cx, cy - r * 0.7, cx + r, cy + r * 0.3], fill=fill, outline=outline, width=width)
    draw.polygon(
        [(cx - r, cy - r * 0.1), (cx + r, cy - r * 0.1), (cx, cy + r * 1.1)],
        fill=fill, outline=outline,
    )


def _draw_sunglasses(draw: ImageDraw.ImageDraw, center: tuple, span: float, colorway: tuple):
    """Heart-shaped sunglasses, one of the persona bible's named rotating
    accessories, centered over the eyes. span is the actual measured
    eye-to-eye pixel distance, so lens size tracks real eye spacing rather
    than a generic scale factor."""
    cx, cy = center
    lens_size = span * 0.85
    gap = span * 0.22
    _draw_heart(draw, (cx - gap - lens_size * 0.5, cy), lens_size, colorway)
    _draw_heart(draw, (cx + gap + lens_size * 0.5, cy), lens_size, colorway)
    draw.line(
        [(cx - gap, cy - lens_size * 0.25), (cx + gap, cy - lens_size * 0.25)],
        fill=colorway[1], width=max(int(lens_size * 0.06), 2),
    )


def _draw_necklace(draw: ImageDraw.ImageDraw, center: tuple, span: float, scale: float, colorway: tuple):
    """A strung-pearl necklace arcing across the base of the neck."""
    import math
    cx, cy = center
    fill, outline = colorway
    pearl_r = 13 * scale
    sag = 22 * scale
    count = 9
    for i in range(count):
        t = i / (count - 1)
        x = cx - span / 2 + span * t
        y = cy + sag * math.sin(math.pi * t)
        draw.ellipse(
            [x - pearl_r, y - pearl_r, x + pearl_r, y + pearl_r],
            fill=fill, outline=outline, width=max(int(pearl_r * 0.3), 1),
        )


def _draw_handbag(draw: ImageDraw.ImageDraw, center: tuple, scale: float, colorway: tuple):
    """A small clutch handbag, positioned near a fist."""
    cx, cy = center
    fill, outline = colorway
    w, h = 78 * scale, 60 * scale
    width = max(int(5 * scale), 2)
    draw.rounded_rectangle(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=10 * scale,
        fill=fill, outline=outline, width=width,
    )
    draw.arc(
        [cx - w * 0.35, cy - h * 1.3, cx + w * 0.35, cy - h * 0.2],
        start=200, end=340, fill=outline, width=width,
    )
    clasp_r = 7 * scale
    draw.ellipse(
        [cx - clasp_r, cy - h * 0.18 - clasp_r, cx + clasp_r, cy - h * 0.18 + clasp_r],
        fill=outline,
    )


def _render_background(size: tuple, base_color: tuple, style: str) -> Image.Image:
    """Builds the canvas background in one of several rotating styles so
    posts don't all read as the same flat color card."""
    w, h = size

    def _shade(color, delta):
        return tuple(max(0, min(255, c + delta)) for c in color)

    if style == "vertical_gradient":
        grad = Image.linear_gradient("L").rotate(90, expand=True).resize((w, h))
        top = Image.new("RGB", (w, h), _shade(base_color, 35))
        bottom = Image.new("RGB", (w, h), _shade(base_color, -35))
        return Image.composite(bottom, top, grad)

    if style == "radial_spotlight":
        grad = Image.radial_gradient("L").resize((w, h))
        center = Image.new("RGB", (w, h), _shade(base_color, 45))
        edge = Image.new("RGB", (w, h), _shade(base_color, -30))
        return Image.composite(edge, center, grad)

    if style == "diagonal_split":
        canvas = Image.new("RGB", (w, h), _shade(base_color, 20))
        draw = ImageDraw.Draw(canvas)
        draw.polygon([(0, h), (w, 0), (w, h)], fill=_shade(base_color, -25))
        return canvas

    if style == "glitter":
        canvas = Image.new("RGB", (w, h), base_color)
        draw = ImageDraw.Draw(canvas)
        sparkle = _shade(base_color, 90)
        for _ in range(60):
            x, y = random.randint(0, w), random.randint(0, h)
            r = random.uniform(1.5, 4)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=sparkle)
        return canvas

    return Image.new("RGB", (w, h), base_color)  # solid


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
    look = random.choices(REFERENCE_LOOKS, weights=[l["weight"] for l in REFERENCE_LOOKS])[0]
    source = Image.open(look["path"]).convert("RGB")
    source_bg_color = source.getpixel((5, 5))
    cutout = _cutout_character(source)

    # Jitter the pillar's base color a little each run and vary the
    # character's size/orientation slightly -- keeps posts visually distinct
    # from each other instead of every "Diva declarations" post looking
    # identical to the last.
    base_color = PILLAR_BACKGROUND_COLORS.get(pillar, source_bg_color)
    canvas_color = tuple(max(0, min(255, c + random.randint(-18, 18))) for c in base_color)
    canvas = _render_background((VIDEO_WIDTH, VIDEO_HEIGHT), canvas_color, random.choice(BACKGROUND_STYLES))
    top_margin = 360  # reserved for text; character fits below this

    available_h = VIDEO_HEIGHT - top_margin
    scale = min(VIDEO_WIDTH / cutout.width, available_h / cutout.height)
    scale *= random.uniform(0.94, 1.04)
    resized = cutout.resize((int(cutout.width * scale), int(cutout.height * scale)), Image.LANCZOS)

    # Draw the bow directly onto the character sprite (not the canvas) so it
    # flips/rotates as one piece with her, in a colorway that rotates per post.
    bow_anchor = look["bow_anchor"]
    bow_local = (bow_anchor[0] * resized.width, bow_anchor[1] * resized.height)
    sprite_draw = ImageDraw.Draw(resized)
    bow_scale = (resized.width / 742) * random.uniform(0.85, 1.15)
    shared_colorway = random.choice(BOW_COLORWAYS)
    _draw_bow(sprite_draw, bow_local, scale=bow_scale, colorway=shared_colorway)

    # One rotating "hero prop" on top of the always-on bow -- coordinated to
    # the same colorway (except pearls, which are always cream) so it reads
    # as a styled look rather than a random grab-bag of props.
    accessory = random.choice(ACCESSORY_POOL)
    accessory_scale = (resized.width / 742) * random.uniform(0.9, 1.1)
    if accessory == "sunglasses":
        eyes_anchor = look["eyes_anchor"]
        eyes_local = (eyes_anchor[0] * resized.width, eyes_anchor[1] * resized.height)
        eyes_span = look["eyes_span"] * resized.width
        _draw_sunglasses(sprite_draw, eyes_local, span=eyes_span, colorway=shared_colorway)
    elif accessory == "necklace":
        necklace_anchor = look["necklace_anchor"]
        necklace_local = (necklace_anchor[0] * resized.width, necklace_anchor[1] * resized.height)
        necklace_span = look["necklace_span"] * resized.width
        _draw_necklace(sprite_draw, necklace_local, span=necklace_span, scale=accessory_scale, colorway=PEARL_COLORWAY)
    elif accessory == "handbag":
        hand_anchor = look["hand_anchor"]
        hand_local = (hand_anchor[0] * resized.width, hand_anchor[1] * resized.height)
        _draw_handbag(sprite_draw, hand_local, scale=accessory_scale, colorway=shared_colorway)

    if random.random() < 0.5:
        resized = resized.transpose(Image.FLIP_LEFT_RIGHT)

    sprite = resized.rotate(random.uniform(-4, 4), resample=Image.BICUBIC, expand=True)

    paste_x = (VIDEO_WIDTH - sprite.width) // 2
    # expand=True pads the bounding box evenly, so re-anchor using the pre-
    # rotation height to keep her feet roughly where they'd land unrotated.
    paste_y = VIDEO_HEIGHT - resized.height - (sprite.height - resized.height) // 2
    canvas.paste(sprite, (paste_x, paste_y), mask=sprite)

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


def _audio_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def build_video(image_path: Path) -> Path:
    """Loops the text-card image for the full length of a random track from
    music/ (capped at Instagram's 90s Reels limit) -- entirely local via
    ffmpeg, no Blotato credits."""
    tracks = sorted(MUSIC_DIR.glob("*.mp3"))
    if not tracks:
        raise RuntimeError(
            f"No music tracks found in {MUSIC_DIR} -- add at least one .mp3 "
            "(see music/README.md)."
        )
    music_path = random.choice(tracks)
    duration = min(_audio_duration_seconds(music_path), MAX_VIDEO_DURATION_SECONDS)

    out_path = image_path.parent / "final.mp4"
    frame_count = int(duration * VIDEO_FPS)
    # Subtle Ken Burns motion so the still image reads as a video rather than
    # a static photo -- the motion style rotates per post (plain zoom-in,
    # zoom-in-with-drift, zoom-out) instead of the same pan every time. Scale
    # up first so zoompan doesn't introduce its own upscale artifacts.
    center_x, center_y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    drift = f"+(on/{frame_count})*50"
    motion = random.choice(ZOOMPAN_MOTION_PRESETS)
    if motion == "zoom_in":
        z_expr, x_expr = "min(zoom+0.0006,1.08)", center_x
    elif motion == "zoom_in_pan_right":
        z_expr, x_expr = "min(zoom+0.0006,1.08)", center_x + drift
    elif motion == "zoom_in_pan_left":
        z_expr, x_expr = "min(zoom+0.0006,1.08)", center_x + drift.replace("+", "-")
    else:  # zoom_out
        z_expr, x_expr = "if(eq(on,0),1.08,max(zoom-0.0006,1.0))", center_x
    zoompan = (
        f"scale={VIDEO_WIDTH * 2}:{VIDEO_HEIGHT * 2},"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{center_y}':d={frame_count}:"
        f"s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}"
    )
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(image_path),
                "-i", str(music_path),
                "-t", str(duration),
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


def _build_target(platform: str, title: str) -> dict:
    if platform == "instagram":
        return {"targetType": "instagram", "mediaType": "reel"}
    if platform == "tiktok":
        return {
            "targetType": "tiktok",
            "privacyLevel": "PUBLIC_TO_EVERYONE",
            "disabledComments": False,
            "disabledDuet": False,
            "disabledStitch": False,
            "isBrandedContent": False,
            "isYourBrand": False,
            "isAiGenerated": True,  # accurate -- this pipeline is fully automated
        }
    if platform == "youtube":
        return {
            "targetType": "youtube",
            "title": title[:90],
            "privacyStatus": "public",
            "shouldNotifySubscribers": True,
            "isMadeForKids": False,
        }
    if platform == "facebook":
        return {"targetType": "facebook", "pageId": BLOTATO_FACEBOOK_PAGE_ID}
    raise ValueError(f"No target builder for platform: {platform}")


def publish_to_platform(platform: str, caption: str, video_url: str, title: str) -> dict:
    resp = requests.post(
        "https://backend.blotato.com/v2/posts",
        headers={
            "blotato-api-key": BLOTATO_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "post": {
                "accountId": BLOTATO_ACCOUNT_IDS[platform],
                "content": {
                    "text": caption,
                    "mediaUrls": [video_url],
                    "platform": platform,
                },
                "target": _build_target(platform, title),
            }
        },
        timeout=30,
    )
    _raise_with_body(resp)
    return resp.json()


def publish_everywhere(caption: str, video_url: str, title: str) -> dict:
    """Publishes to every connected platform concurrently. A failure on one
    platform doesn't block the others -- each result/error is reported
    separately so a partial post (e.g. Instagram succeeds, TikTok rejects the
    video) is still visible rather than silently lost."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(BLOTATO_ACCOUNT_IDS)) as pool:
        futures = {
            pool.submit(publish_to_platform, platform, caption, video_url, title): platform
            for platform in BLOTATO_ACCOUNT_IDS
        }
        for future in as_completed(futures):
            platform = futures[future]
            try:
                results[platform] = {"ok": True, "result": future.result()}
            except Exception as e:
                results[platform] = {"ok": False, "error": str(e)}
    return results


def append_log(entry: dict):
    text = PERSONA_PATH.read_text()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    line = (
        f"- {date} | pillar: {entry['pillar_used']} | "
        f"format: {entry.get('format_used') or 'unspecified'} | "
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

    # YouTube needs a distinct title; the text-card line doubles as one, with
    # line breaks flattened since it was written to be read across lines, not
    # as a single sentence.
    title = " ".join(post["image_text"].split("\n")).strip()
    results = publish_everywhere(post["caption"], video_url, title)

    failures = {platform: r["error"] for platform, r in results.items() if not r["ok"]}
    if failures:
        notify_owner(f"Publish failed on {len(failures)} platform(s): {failures}")
    if len(failures) == len(results):
        return  # nothing published anywhere -- don't record the log entry

    append_log(post)
    print(f"Published: {results}")


if __name__ == "__main__":
    main()
