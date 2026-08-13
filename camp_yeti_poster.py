"""
Camp Yeti autonomous poster.
Reads the persona bible, asks Claude for a caption + a short text-card line,
renders it as an animated video of the fixed reference character (no AI
image/video generation -- same approved artwork every time, brought to life
with a local beat-synced groove/blink animation instead), sets it to a
full-length music track, and publishes it to Instagram and Facebook.

The video is hosted as a GitHub Release asset on this repo (free, no
external storage account) to get a public URL, since Instagram and
Facebook's publish APIs need to fetch the file from a URL rather than
accept a direct upload. Both publish directly via the Meta Graph API --
no third-party posting service involved.

Env vars required:
  ANTHROPIC_API_KEY
  GITHUB_TOKEN                  (auto-provided by Actions -- must be passed
                                  through explicitly in the workflow's env block)
  META_PAGE_ACCESS_TOKEN        (long-lived Page token: pages_manage_posts,
                                  pages_read_engagement, instagram_content_publish)
  META_IG_BUSINESS_ID           (Instagram Business Account ID, linked to the Page)
  META_PAGE_ID                  (Facebook Page ID)

Also requires ffmpeg on PATH, and `pip install Pillow`.

Run this on a schedule (see .github/workflows/camp_yeti_post.yml).
"""

import os
import json
import math
import re
import sys
import time
import bisect
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import librosa
except ImportError:
    librosa = None  # beat-synced dancing degrades to a plain breathing bob if unavailable

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Instagram + Facebook publish directly via the Meta Graph API.
META_PAGE_ACCESS_TOKEN = os.environ["META_PAGE_ACCESS_TOKEN"]
META_IG_BUSINESS_ID = os.environ["META_IG_BUSINESS_ID"]
META_PAGE_ID = os.environ["META_PAGE_ID"]
META_GRAPH_VERSION = "v20.0"

# For hosting the video as a public GitHub Release asset (see
# upload_video_to_github_release). GITHUB_REPOSITORY ("owner/repo") is set
# automatically by Actions; GITHUB_TOKEN must be passed through explicitly
# in the workflow's env block since Actions doesn't inject it by default.
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]

PERSONA_PATH = Path(__file__).parent / "camp_yeti_persona_bible.md"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "camp_yeti_agent_system_prompt.md"
REFERENCE_DIR = Path(__file__).parent / "reference"
FONT_PATH = Path(__file__).parent / "fonts" / "Anton-Regular.ttf"
MUSIC_DIR = Path(__file__).parent / "music"
POSTING_STATE_PATH = Path(__file__).parent / "posting_state.json"

# Posting cadence alternates 2-day and 3-day gaps (not a fixed interval) to
# build suspense/unpredictability rather than posting like clockwork. The
# workflow's cron fires daily; main() below decides whether today is
# actually a posting day.
POSTING_GAP_DAYS = (2, 3)

ANTHROPIC_MODEL = "claude-sonnet-4-6"

VIDEO_WIDTH, VIDEO_HEIGHT = 1080, 1350  # 4:5 -- works for both feed and reel
MAX_VIDEO_DURATION_SECONDS = 300  # absolute safety ceiling only
TARGET_REEL_DURATION_SECONDS = 16  # default length for most posts -- short and loopable outperforms full-track length on completion rate, one of Reels' strongest ranking signals
FULL_LENGTH_POST_CHANCE = 0.2  # roughly 1 in 5 posts instead plays the full original track as a deliberate showcase piece
AUDIO_FADE_OUT_SECONDS = 1.0
VIDEO_FPS = 25
ANIMATION_FPS = 12  # frame-generation rate for the character animation -- smooth enough for a subtle bob/blink, cheaper to render than full VIDEO_FPS; ffmpeg upsamples to VIDEO_FPS on output
FONT_SIZE = 72
TEXT_COLOR = (255, 255, 255)
TEXT_OUTLINE_COLOR = (20, 20, 40)

# Character animation, so posts read as actual video rather than a still
# image with a camera pan over it. Primary mode is a beat-synced groove
# (bounce + alternating side sway) timed to the actual track via librosa
# beat detection; falls back to a plain sine-wave breathing bob if librosa
# is unavailable or beat detection fails on a given track. Blinking runs
# either way.
BOB_AMPLITUDE_PX = 7  # fallback-mode amplitude
BOB_PERIOD_SECONDS = 3.0  # fallback-mode period
BOUNCE_AMPLITUDE_PX = 24  # beat-mode vertical bounce
SWAY_AMPLITUDE_PX = 18  # beat-mode side-to-side sway
BLINK_DURATION_SECONDS = 0.15
BLINK_INTERVAL_RANGE = (1.8, 4.2)  # seconds between blinks, randomized per blink
DOUBLE_BLINK_CHANCE = 0.25  # chance a given blink is followed by a quick second one
DOUBLE_BLINK_GAP_SECONDS = 0.22

# Shimmer: a soft diagonal light sweep across the character, masked to her
# own silhouette (via sprite alpha) so it reads as a shine/highlight rather
# than a generic overlay. Cycles continuously through the video.
SHIMMER_PERIOD_SECONDS = 4.0
SHIMMER_BAND_WIDTH_FRAC = 0.22  # fraction of sprite width
SHIMMER_PEAK_ALPHA = 130

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

# Ken Burns camera-motion presets for build_animated_video -- rotates so
# posts don't all pan the exact same way.
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
        "eyes_anchor": (0.573, 0.207),  # midpoint between the eyes, for sunglasses -- the reference art is a 3/4 turned pose, so the eyes sit much closer together in screen space than a front-facing view would suggest
        "eyes_span": 0.068,  # fraction of width between the two eyes, for sizing (measured directly from pixel data, not assumed symmetry)
        "left_eye_anchor": (0.539, 0.202),  # for blink animation -- measured individually since the 3/4 pose isn't symmetric
        "right_eye_anchor": (0.608, 0.212),
        "mouth_anchor": (0.674, 0.268),  # center of the mouth opening, for the sing/roar animation
        "mouth_span": 0.47,  # fraction of width across the mouth opening
        "ear_anchor": (0.445, 0.21),  # visible ear, for a dangling earring
        "hat_anchor": (0.568, 0.11),  # crest tip (measured from actual pixel data), for a tilted party hat -- clear of the bow's spot at the base of the crest
        "hand_anchor": (0.19, 0.77),  # left fist, for a handbag or rings
        "weight": 1,
    },
]

# One rotating "hero prop" per the persona bible's accessory rule (never more
# than one at once, on top of the always-on bow). Weighted toward "none" so
# it stays occasional rather than cluttering the silhouette every time.
# Pearls/necklace were retired -- they read as oddly placed no matter how
# precisely they were positioned, so they're gone rather than fought further.
ACCESSORY_POOL = ["none", "none", "sunglasses", "handbag", "earrings", "rings", "hat"]

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
    "chant hook -- short, rhythmic, repeatable phrase built to be chanted/danced to, high camp energy, like a crowd could shout it back; still an original line in her voice, not a reference to any existing song",
]


def generate_post(persona_bible: str, theme_hint: str = None) -> dict:
    """Ask Claude for a JSON-structured post: caption + short on-image text-card line.

    theme_hint is an optional short, original mood/theme description (never
    the track's actual lyrics -- those aren't passed to the model) used to
    loosely inspire this post when it's set to a specific song, without
    quoting or echoing the song's own words."""
    formats_block = "\n".join(f"- {f}" for f in CAPTION_FORMATS)
    system = (
        "You are the autonomous content generator for the Camp Yeti Instagram "
        "persona. Follow the persona bible exactly. Respond with ONLY valid JSON, "
        "no markdown fences, no preamble, matching this schema:\n"
        '{"caption": "...", "image_text": "...", "pillar_used": "...", '
        '"format_used": "...", "phrase_used": "... or null", "new_lore": "... or null"}'
    )
    theme_block = (
        f"\nThis post is set to a track with this general mood: {theme_hint}. "
        "Let that mood loosely color the pillar/format/energy you pick, but "
        "write entirely original lines in Yeti's own voice -- don't quote, "
        "paraphrase closely, or echo any specific words/phrases from the "
        "track itself.\n" if theme_hint else ""
    )
    user = (
        f"PERSONA BIBLE:\n{persona_bible}\n"
        f"{theme_block}\n"
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


def _draw_earring(draw: ImageDraw.ImageDraw, center: tuple, scale: float, colorway: tuple):
    """A single dangling earring hanging from the visible ear -- a small
    stud plus a teardrop below it."""
    cx, cy = center
    fill, outline = colorway
    stud_r = 7 * scale
    width = max(int(stud_r * 0.35), 2)
    draw.ellipse([cx - stud_r, cy - stud_r, cx + stud_r, cy + stud_r], fill=fill, outline=outline, width=width)
    drop_w, drop_h = 10 * scale, 20 * scale
    drop_y = cy + stud_r * 1.4
    draw.polygon(
        [(cx - drop_w / 2, drop_y), (cx + drop_w / 2, drop_y), (cx, drop_y + drop_h)],
        fill=fill, outline=outline,
    )
    draw.ellipse(
        [cx - drop_w / 2, drop_y - drop_w / 4, cx + drop_w / 2, drop_y + drop_w / 2],
        fill=fill, outline=outline, width=width,
    )


def _draw_rings(draw: ImageDraw.ImageDraw, center: tuple, scale: float, colorway: tuple):
    """A chunky bangle wrapping the wrist -- rings on individual fingers
    don't read on a fist this stylized (no separate fingers drawn), a band
    around the wrist does."""
    cx, cy = center
    fill, outline = colorway
    w, h = 46 * scale, 30 * scale
    width = max(int(7 * scale), 3)
    draw.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=outline, width=width)
    draw.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h * 0.05], fill=fill, outline=outline, width=width)


def _draw_hat(draw: ImageDraw.ImageDraw, center: tuple, scale: float, colorway: tuple):
    """A small tilted party hat perched at the crest tip, clear of the
    bow's spot at the base of the crest."""
    cx, cy = center
    fill, outline = colorway
    width = max(int(7 * scale), 3)
    w, h = 60 * scale, 60 * scale
    tip = (cx + 10 * scale, cy - h)
    draw.polygon(
        [(cx - w / 2, cy + h * 0.1), (cx + w / 2, cy + h * 0.1), tip],
        fill=fill, outline=outline, width=width,
    )
    pom_r = 13 * scale
    draw.ellipse([tip[0] - pom_r, tip[1] - pom_r, tip[0] + pom_r, tip[1] + pom_r], fill=outline, outline=outline)
    draw.ellipse(
        [cx - w / 2 - 4 * scale, cy - h * 0.02, cx + w / 2 + 4 * scale, cy + h * 0.22],
        fill=fill, outline=outline, width=width,
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


def _apply_shimmer(sprite: Image.Image, t: float) -> Image.Image:
    """Sweeps a soft diagonal band of light across the character, masked to
    her own silhouette (via the sprite's alpha channel) so it reads as a
    shimmer/highlight on her fur rather than a generic overlay drifting over
    the whole frame. `t` is the current time in seconds; the sweep loops
    every SHIMMER_PERIOD_SECONDS."""
    if sprite.mode != "RGBA":
        sprite = sprite.convert("RGBA")
    w, h = sprite.size
    phase = (t % SHIMMER_PERIOD_SECONDS) / SHIMMER_PERIOD_SECONDS
    band_width = w * SHIMMER_BAND_WIDTH_FRAC
    center_x = -band_width + phase * (w + 2 * band_width)

    band_mask = Image.new("L", (w, h), 0)
    band_draw = ImageDraw.Draw(band_mask)
    steps = 14
    for i in range(steps):
        frac = i / (steps - 1)
        alpha = int(SHIMMER_PEAK_ALPHA * (1 - abs(frac - 0.5) * 2))
        x_off = center_x + (frac - 0.5) * band_width
        band_draw.line([(x_off, 0), (x_off - h * 0.35, h)], fill=alpha, width=4)

    sprite_alpha = sprite.split()[-1]
    shimmer_mask = Image.composite(band_mask, Image.new("L", (w, h), 0), sprite_alpha)
    shimmer_layer = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    shimmer_layer.putalpha(shimmer_mask)
    return Image.alpha_composite(sprite, shimmer_layer)


def _draw_character_shadow(canvas: Image.Image, sprite_width: int, sprite_height: int, paste_x: int, paste_y: int):
    """Soft ellipse shadow at her feet so she reads as grounded on the
    background rather than pasted flat on top of it."""
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    feet_y = paste_y + sprite_height - 24
    shadow_w = sprite_width * 0.55
    shadow_x = paste_x + (sprite_width - shadow_w) / 2
    draw.ellipse([shadow_x, feet_y, shadow_x + shadow_w, feet_y + 36], fill=(0, 0, 0, 75))
    blurred = overlay.filter(ImageFilter.GaussianBlur(8))
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), blurred).convert("RGB"), (0, 0))


def _draw_text_backdrop(canvas: Image.Image, y_start: int, y_end: int):
    """Soft translucent dark bar behind the text block so captions stay
    readable regardless of which background style got rolled (diagonal
    splits and glitter in particular can eat into contrast)."""
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, max(y_start - 28, 0), canvas.width, y_end + 20], fill=(12, 12, 22, 95))
    blurred = overlay.filter(ImageFilter.GaussianBlur(2))
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), blurred).convert("RGB"), (0, 0))


def _draw_blink(draw: ImageDraw.ImageDraw, center: tuple, eye_span: float, skin_color: tuple):
    """Draws a closed eyelid over one eye -- a simple curved line filled
    with the sampled local skin tone, so it reads as the eye shutting
    rather than a patch pasted over the face."""
    cx, cy = center
    w, h = eye_span * 0.62, eye_span * 0.22
    draw.pieslice(
        [cx - w / 2, cy - h, cx + w / 2, cy + h], start=15, end=165,
        fill=skin_color, outline=(30, 40, 55), width=max(int(eye_span * 0.06), 2),
    )


def _draw_mouth_open(draw: ImageDraw.ImageDraw, center: tuple, span: float):
    """Drops the jaw open wider (purple-pink interior, per the persona
    bible's roar/shout expression) so she reads as singing/shouting along
    on the beat, not just standing there. Sized to extend the existing
    mouth line rather than overwhelm it."""
    cx, cy = center
    w, h = span * 0.26, span * 0.15
    draw.ellipse(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        fill=(90, 30, 65), outline=(30, 40, 55), width=max(int(span * 0.02), 2),
    )
    tongue_w, tongue_h = w * 0.45, h * 0.5
    draw.ellipse(
        [cx - tongue_w / 2, cy, cx + tongue_w / 2, cy + tongue_h],
        fill=(190, 85, 120),
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


def _build_scene(image_text: str, pillar: str = None) -> dict:
    """Builds everything a post needs except the final per-frame compositing:
    the background+text canvas (identical for every frame of the video), and
    two character sprite variants -- eyes open and eyes closed -- flipped and
    rotated identically, so a renderer can swap between them for a blink and
    translate either vertically for a breathing bob. Entirely local, no AI
    image generation, so the art is always the same approved portrait.

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
    # the same colorway so it reads as a styled look rather than a random
    # grab-bag of props.
    accessory = random.choice(ACCESSORY_POOL)
    accessory_scale = (resized.width / 742) * random.uniform(0.9, 1.1)
    if accessory == "sunglasses":
        eyes_anchor = look["eyes_anchor"]
        eyes_local = (eyes_anchor[0] * resized.width, eyes_anchor[1] * resized.height)
        eyes_span = look["eyes_span"] * resized.width
        _draw_sunglasses(sprite_draw, eyes_local, span=eyes_span, colorway=shared_colorway)
    elif accessory == "handbag":
        hand_anchor = look["hand_anchor"]
        hand_local = (hand_anchor[0] * resized.width, hand_anchor[1] * resized.height)
        _draw_handbag(sprite_draw, hand_local, scale=accessory_scale, colorway=shared_colorway)
    elif accessory == "earrings":
        ear_anchor = look["ear_anchor"]
        ear_local = (ear_anchor[0] * resized.width, ear_anchor[1] * resized.height)
        _draw_earring(sprite_draw, ear_local, scale=accessory_scale, colorway=shared_colorway)
    elif accessory == "rings":
        hand_anchor = look["hand_anchor"]
        # Offset up from the fist toward the wrist -- a band wraps the
        # wrist cleanly, unlike individual finger rings on a fist this
        # stylized.
        wrist_local = (hand_anchor[0] * resized.width, (hand_anchor[1] - 0.05) * resized.height)
        _draw_rings(sprite_draw, wrist_local, scale=accessory_scale, colorway=shared_colorway)
    elif accessory == "hat":
        hat_anchor = look["hat_anchor"]
        hat_local = (hat_anchor[0] * resized.width, hat_anchor[1] * resized.height)
        _draw_hat(sprite_draw, hat_local, scale=accessory_scale, colorway=shared_colorway)

    # A second sprite variant with both eyes closed, for the blink. Skipped
    # when sunglasses are the chosen accessory -- the eyes are already
    # hidden, so a blink underneath opaque lenses would never be visible.
    resized_blink = resized.copy()
    if accessory != "sunglasses":
        blink_draw = ImageDraw.Draw(resized_blink)
        skin_sample_x = int(look["left_eye_anchor"][0] * resized.width)
        skin_sample_y = max(0, int(look["left_eye_anchor"][1] * resized.height) - 16)
        skin_color = resized.convert("RGB").getpixel((skin_sample_x, skin_sample_y))
        eyes_span_px = look["eyes_span"] * resized.width
        for eye_key in ("left_eye_anchor", "right_eye_anchor"):
            eye_anchor = look[eye_key]
            eye_local = (eye_anchor[0] * resized.width, eye_anchor[1] * resized.height)
            _draw_blink(blink_draw, eye_local, eye_span=eyes_span_px, skin_color=skin_color)

    # A third variant with the mouth dropped open wider, so she can look
    # like she's singing/shouting along on the beat.
    resized_singing = resized.copy()
    singing_draw = ImageDraw.Draw(resized_singing)
    mouth_anchor = look["mouth_anchor"]
    mouth_local = (mouth_anchor[0] * resized.width, mouth_anchor[1] * resized.height)
    mouth_span_px = look["mouth_span"] * resized.width
    _draw_mouth_open(singing_draw, mouth_local, span=mouth_span_px)

    flip = random.random() < 0.5
    if flip:
        resized = resized.transpose(Image.FLIP_LEFT_RIGHT)
        resized_blink = resized_blink.transpose(Image.FLIP_LEFT_RIGHT)
        resized_singing = resized_singing.transpose(Image.FLIP_LEFT_RIGHT)

    rotation = random.uniform(-4, 4)
    sprite_open = resized.rotate(rotation, resample=Image.BICUBIC, expand=True)
    sprite_blink = resized_blink.rotate(rotation, resample=Image.BICUBIC, expand=True)
    sprite_singing = resized_singing.rotate(rotation, resample=Image.BICUBIC, expand=True)

    paste_x = (VIDEO_WIDTH - sprite_open.width) // 2
    # expand=True pads the bounding box evenly, so re-anchor using the pre-
    # rotation height to keep her feet roughly where they'd land unrotated.
    paste_y = VIDEO_HEIGHT - resized.height - (sprite_open.height - resized.height) // 2

    _draw_character_shadow(canvas, sprite_open.width, sprite_open.height, paste_x, paste_y)

    draw = ImageDraw.Draw(canvas)

    # Font size responds to how much text there is -- a punchy one-liner
    # gets to be bigger and bolder; a longer rule-of-three build shrinks a
    # touch so it still fits cleanly above the character.
    line_count_estimate = len([l for l in image_text.split("\n") if l.strip()])
    if line_count_estimate <= 1:
        dynamic_font_size = FONT_SIZE + 22
    elif line_count_estimate <= 3:
        dynamic_font_size = FONT_SIZE
    else:
        dynamic_font_size = FONT_SIZE - 12
    font = ImageFont.truetype(str(FONT_PATH), dynamic_font_size)

    lines = []
    for raw_line in image_text.split("\n"):
        lines.extend(_wrap_text(draw, raw_line.upper(), font, VIDEO_WIDTH - 120))

    line_height = dynamic_font_size + 16
    total_text_height = line_height * len(lines)
    y = max((top_margin - total_text_height) // 2, 40)

    _draw_text_backdrop(canvas, y, y + total_text_height)
    draw = ImageDraw.Draw(canvas)  # re-acquire after the backdrop paste replaced the canvas image

    for line in lines:
        width = draw.textlength(line, font=font)
        x = (VIDEO_WIDTH - width) / 2
        for dx, dy in [(-3, -3), (-3, 3), (3, -3), (3, 3), (-3, 0), (3, 0), (0, -3), (0, 3)]:
            draw.text((x + dx, y + dy), line, font=font, fill=TEXT_OUTLINE_COLOR)
        draw.text((x, y), line, font=font, fill=TEXT_COLOR)
        y += line_height

    return {
        "canvas": canvas,
        "sprite_open": sprite_open,
        "sprite_blink": sprite_blink,
        "sprite_singing": sprite_singing,
        "paste_x": paste_x,
        "paste_y": paste_y,
    }


def render_text_card_image(image_text: str, pillar: str = None) -> Path:
    """Standalone still-image render (used for quick local previews/tests) --
    the live pipeline uses _build_scene + build_animated_video directly."""
    scene = _build_scene(image_text, pillar)
    canvas = scene["canvas"].copy()
    sprite = _apply_shimmer(scene["sprite_open"], t=SHIMMER_PERIOD_SECONDS * 0.4)  # a pleasant fixed highlight position for stills
    canvas.paste(sprite, (scene["paste_x"], scene["paste_y"]), mask=sprite)
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


def _detect_beats(path: Path) -> list:
    """Detects beat timestamps in the actual track via librosa, so the
    character can bounce/sway in time with the real rhythm rather than a
    generic fixed period. Returns [] if librosa is unavailable or detection
    fails on this track -- callers fall back to a plain sine bob."""
    if librosa is None:
        return []
    try:
        y, sr = librosa.load(str(path), sr=22050)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        return list(librosa.frames_to_time(beat_frames, sr=sr))
    except Exception:
        return []


def _groove_offset(t: float, beat_times: list) -> tuple:
    """Beat-synced motion at time t: a decaying vertical bounce right after
    each beat, plus a side-to-side sway that alternates direction beat to
    beat -- a whole-body groove (the character art is a single flat
    illustration, not separate limb layers, so true independent arm/leg
    movement isn't achievable without new source art)."""
    idx = bisect.bisect_right(beat_times, t) - 1
    if idx < 0:
        return 0.0, 0.0
    beat_t = beat_times[idx]
    next_t = beat_times[idx + 1] if idx + 1 < len(beat_times) else beat_t + 0.6
    interval = max(next_t - beat_t, 0.05)
    phase = min((t - beat_t) / interval, 1.0)
    bounce = BOUNCE_AMPLITUDE_PX * math.exp(-phase * 5) * math.sin(phase * math.pi)
    sway_sign = 1 if idx % 2 == 0 else -1
    sway = SWAY_AMPLITUDE_PX * sway_sign * math.sin(phase * math.pi)
    return bounce, sway


def _apply_zoom(frame: Image.Image, zoom: float, pan_x: float) -> Image.Image:
    """Digital zoom/pan -- upscale then crop back to frame size, centered
    plus an optional horizontal drift. Mirrors the old ffmpeg zoompan
    motion, computed directly in Python so it composites cleanly with the
    per-frame character animation instead of fighting a second filter stage."""
    if zoom <= 1.0 and pan_x == 0:
        return frame
    w, h = frame.size
    upscaled = frame.resize((max(w, int(w * zoom)), max(h, int(h * zoom))), Image.BILINEAR)
    cx = (upscaled.width - w) / 2 + pan_x
    cy = (upscaled.height - h) / 2
    cx = max(0, min(upscaled.width - w, cx))
    cy = max(0, min(upscaled.height - h, cy))
    return upscaled.crop((int(cx), int(cy), int(cx) + w, int(cy) + h))


def build_animated_video(image_text: str, pillar: str = None, music_path: Path = None) -> Path:
    """Renders Camp Yeti as actual moving video -- a beat-synced groove
    (bounce + alternating sway, timed to the real track via librosa beat
    detection) and periodic blinks on the character, plus a rotating Ken
    Burns-style camera drift, composited frame-by-frame in Python and piped
    straight to ffmpeg against a full-length music track. Still no AI
    generation involved.

    music_path optionally pins a specific track instead of a random one --
    used when a post is deliberately paired with a particular song."""
    scene = _build_scene(image_text, pillar)
    canvas = scene["canvas"]
    sprite_open = scene["sprite_open"]
    sprite_blink = scene["sprite_blink"]
    sprite_singing = scene["sprite_singing"]
    paste_x, paste_y = scene["paste_x"], scene["paste_y"]

    if music_path is None:
        tracks = sorted(MUSIC_DIR.glob("*.mp3"))
        if not tracks:
            raise RuntimeError(
                f"No music tracks found in {MUSIC_DIR} -- add at least one .mp3 "
                "(see music/README.md)."
            )
        music_path = random.choice(tracks)
    full_track_duration = _audio_duration_seconds(music_path)
    is_full_length_post = random.random() < FULL_LENGTH_POST_CHANCE
    if is_full_length_post:
        duration = min(full_track_duration, MAX_VIDEO_DURATION_SECONDS)
    else:
        duration = min(full_track_duration, TARGET_REEL_DURATION_SECONDS)
    # Only fade the audio out early if we're actually cutting it short --
    # a full-length showcase post should just let the track end naturally.
    is_trimmed = duration < full_track_duration - 0.05
    frame_count = int(duration * ANIMATION_FPS)
    motion = random.choice(ZOOMPAN_MOTION_PRESETS)
    beat_times = _detect_beats(music_path)

    # Schedule blinks across the whole track up front as (start, end) second
    # windows, rather than re-rolling randomness inside the frame loop.
    blink_windows = []
    t_cursor = random.uniform(*BLINK_INTERVAL_RANGE)
    while t_cursor < duration:
        blink_windows.append((t_cursor, t_cursor + BLINK_DURATION_SECONDS))
        if random.random() < DOUBLE_BLINK_CHANCE:
            second_start = t_cursor + BLINK_DURATION_SECONDS + DOUBLE_BLINK_GAP_SECONDS
            blink_windows.append((second_start, second_start + BLINK_DURATION_SECONDS))
            t_cursor = second_start + BLINK_DURATION_SECONDS
        t_cursor += random.uniform(*BLINK_INTERVAL_RANGE)

    out_path = Path(tempfile.mkdtemp(prefix="camp-yeti-")) / "final.mp4"

    audio_filter_args = []
    if is_trimmed:
        fade_start = max(duration - AUDIO_FADE_OUT_SECONDS, 0)
        audio_filter_args = ["-af", f"afade=t=out:st={fade_start}:d={AUDIO_FADE_OUT_SECONDS}"]

    try:
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}", "-r", str(ANIMATION_FPS),
                "-i", "-",
                "-i", str(music_path),
                "-t", str(duration),
                *audio_filter_args,
                "-r", str(VIDEO_FPS),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                str(out_path),
            ],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH.")

    try:
        for i in range(frame_count):
            t = i / ANIMATION_FPS
            if beat_times:
                bounce, sway = _groove_offset(t, beat_times)
            else:
                bounce = BOB_AMPLITUDE_PX * math.sin(2 * math.pi * t / BOB_PERIOD_SECONDS)
                sway = 0.0
            blinking = any(start <= t <= end for start, end in blink_windows)
            # Mouth-open/singing sync is built (sprite_singing, _draw_mouth_open)
            # but disabled for now -- it didn't blend cleanly with the jaw
            # line even after fixing a real centering bug, and shipping a
            # visibly-off mouth blob isn't worth it. Revisit with a shape
            # that follows the actual jaw contour instead of a generic oval.
            singing = False

            if blinking:
                sprite = sprite_blink
            elif singing:
                sprite = sprite_singing
            else:
                sprite = sprite_open

            sprite = _apply_shimmer(sprite, t)

            frame = canvas.copy()
            frame.paste(sprite, (paste_x + int(sway), paste_y + int(bounce)), mask=sprite)

            t_frac = i / frame_count
            if motion == "zoom_in":
                zoom, pan_x = 1.0 + 0.08 * t_frac, 0
            elif motion == "zoom_in_pan_right":
                zoom, pan_x = 1.0 + 0.08 * t_frac, 60 * t_frac
            elif motion == "zoom_in_pan_left":
                zoom, pan_x = 1.0 + 0.08 * t_frac, -60 * t_frac
            else:  # zoom_out
                zoom, pan_x = 1.08 - 0.08 * t_frac, 0
            frame = _apply_zoom(frame, zoom, pan_x)

            ffmpeg.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass  # surfaced below via the non-zero return code
    finally:
        ffmpeg.stdin.close()
        stderr = ffmpeg.stderr.read().decode(errors="replace")
        ffmpeg.wait()

    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg failed building video: {stderr}")

    return out_path


def upload_video_to_github_release(video_path: Path) -> str:
    """Publishes the video as a GitHub Release asset on this repo and
    returns its public download URL. Free, no external storage account --
    Instagram and Facebook's publish APIs need a URL to fetch from rather
    than a direct upload, and this reuses infrastructure the repo already has."""
    tag = f"post-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    api = f"https://api.github.com/repos/{GITHUB_REPOSITORY}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    release_resp = requests.post(
        f"{api}/releases",
        headers=headers,
        json={"tag_name": tag, "name": tag, "draft": False, "prerelease": False},
        timeout=30,
    )
    _raise_with_body(release_resp)
    upload_url = release_resp.json()["upload_url"].split("{")[0]

    with open(video_path, "rb") as f:
        asset_resp = requests.post(
            f"{upload_url}?name=camp-yeti-post.mp4",
            headers={**headers, "Content-Type": "video/mp4"},
            data=f,
            timeout=120,
        )
    _raise_with_body(asset_resp)
    return asset_resp.json()["browser_download_url"]


def publish_to_instagram_direct(caption: str, video_url: str, cover_offset_ms: int = 1200) -> dict:
    """Publishes a Reel directly via the Instagram Graph API: create a media
    container, poll until Instagram finishes fetching/processing the video,
    then publish the container.

    share_to_feed=True asks Instagram to also surface the Reel in the main
    feed grid (not just the Reels tab) -- without it, distribution can be
    limited to Reels-only surfaces, which meaningfully caps reach.

    cover_offset_ms picks the video's cover/thumbnail frame -- defaults to
    ~1.2s in, since the caption text card is already fully on screen by then
    rather than showing a blank first frame."""
    base = f"https://graph.facebook.com/{META_GRAPH_VERSION}"

    create_resp = requests.post(
        f"{base}/{META_IG_BUSINESS_ID}/media",
        params={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
            "thumb_offset": cover_offset_ms,
            "access_token": META_PAGE_ACCESS_TOKEN,
        },
        timeout=30,
    )
    _raise_with_body(create_resp)
    creation_id = create_resp.json()["id"]

    for _ in range(30):
        status_resp = requests.get(
            f"{base}/{creation_id}",
            params={"fields": "status_code", "access_token": META_PAGE_ACCESS_TOKEN},
            timeout=30,
        )
        _raise_with_body(status_resp)
        status = status_resp.json().get("status_code")
        if status == "FINISHED":
            break
        if status == "ERROR":
            raise RuntimeError(f"Instagram container processing failed: {status_resp.json()}")
        time.sleep(10)
    else:
        raise RuntimeError("Instagram container never finished processing (timed out)")

    publish_resp = requests.post(
        f"{base}/{META_IG_BUSINESS_ID}/media_publish",
        params={"creation_id": creation_id, "access_token": META_PAGE_ACCESS_TOKEN},
        timeout=30,
    )
    _raise_with_body(publish_resp)
    return publish_resp.json()


def publish_to_facebook_direct(caption: str, video_url: str) -> dict:
    """Publishes a video directly to the Facebook Page via the Graph API."""
    base = f"https://graph.facebook.com/{META_GRAPH_VERSION}"
    resp = requests.post(
        f"{base}/{META_PAGE_ID}/videos",
        params={
            "file_url": video_url,
            "description": caption,
            "access_token": META_PAGE_ACCESS_TOKEN,
        },
        timeout=60,
    )
    _raise_with_body(resp)
    return resp.json()


DIRECT_PUBLISHERS = {
    "instagram": publish_to_instagram_direct,
    "facebook": publish_to_facebook_direct,
}


def publish_everywhere(caption: str, video_url: str) -> dict:
    """Publishes to Instagram and Facebook concurrently, both directly via
    the Meta Graph API. A failure on one platform doesn't block the other --
    each result/error is reported separately so a partial post (e.g.
    Instagram succeeds, Facebook rejects the video) is still visible rather
    than silently lost."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(DIRECT_PUBLISHERS)) as pool:
        futures = {
            pool.submit(publisher, caption, video_url): platform
            for platform, publisher in DIRECT_PUBLISHERS.items()
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


def _last_post_date() -> "datetime.date":
    """Parses the continuity log's most recent dated entry."""
    text = PERSONA_PATH.read_text()
    dates = re.findall(r"^- (\d{4}-\d{2}-\d{2}) \|", text, re.MULTILINE)
    if not dates:
        return None
    return datetime.strptime(dates[-1], "%Y-%m-%d").date()


def _load_posting_state() -> dict:
    if POSTING_STATE_PATH.exists():
        return json.loads(POSTING_STATE_PATH.read_text())
    return {"last_gap_days": POSTING_GAP_DAYS[-1]}  # so the very first gap picked is the other value


def _save_posting_state(state: dict):
    POSTING_STATE_PATH.write_text(json.dumps(state, indent=2))


def _is_posting_day() -> bool:
    """Alternates 2-day and 3-day gaps between posts (not a fixed interval)
    to read as less mechanical/predictable -- deliberately building
    suspense rather than posting on a clockwork schedule."""
    last_date = _last_post_date()
    if last_date is None:
        return True  # no history yet -- just post
    days_since = (datetime.now(timezone.utc).date() - last_date).days
    state = _load_posting_state()
    next_gap = POSTING_GAP_DAYS[0] if state.get("last_gap_days") == POSTING_GAP_DAYS[1] else POSTING_GAP_DAYS[1]
    return days_since >= next_gap


def _record_posting_gap():
    """Call only after a successful publish -- flips which gap (2 or 3 days) is due next."""
    last_date = _last_post_date()
    days_since = (datetime.now(timezone.utc).date() - last_date).days if last_date else POSTING_GAP_DAYS[0]
    # Snap to whichever configured gap this run's actual spacing is closest
    # to, so the alternation stays correct even if a run was skipped/delayed.
    closest_gap = min(POSTING_GAP_DAYS, key=lambda g: abs(g - days_since))
    _save_posting_state({"last_gap_days": closest_gap})


def notify_owner(message: str):
    """Wire this to email/Slack/SMS -- whatever reaches you. Placeholder: prints."""
    print(f"[ESCALATION] {message}")
    # e.g. requests.post(SLACK_WEBHOOK_URL, json={"text": message})


def main():
    if not _is_posting_day():
        print("Not a posting day (alternating 2/3-day gap) -- skipping.")
        return

    persona_bible = PERSONA_PATH.read_text()

    for attempt in range(3):
        try:
            post = generate_post(persona_bible)
            break
        except Exception as e:
            if attempt == 2:
                notify_owner(f"Post generation failed 3x: {e}")
                sys.exit(1)  # fail the run visibly -- a silent 'success' with nothing posted is how gaps go unnoticed
            time.sleep(2)

    try:
        video_path = build_animated_video(post["image_text"], pillar=post.get("pillar_used"))
    except Exception as e:
        notify_owner(f"Video assembly failed: {e}")
        sys.exit(1)

    try:
        video_url = upload_video_to_github_release(video_path)
    except Exception as e:
        notify_owner(f"Video upload failed: {e}")
        sys.exit(1)

    results = publish_everywhere(post["caption"], video_url)

    failures = {platform: r["error"] for platform, r in results.items() if not r["ok"]}
    if failures:
        notify_owner(f"Publish failed on {len(failures)} platform(s): {failures}")
    if len(failures) == len(results):
        # Nothing published anywhere -- don't record the log entry, and fail
        # the run so GitHub actually flags it rather than reporting a quiet
        # 'success' with no post to show for it. A partial failure (e.g. the
        # known Facebook issue) still exits 0 since real posts did go out.
        sys.exit(1)

    append_log(post)
    _record_posting_gap()
    print(f"Published: {results}")


if __name__ == "__main__":
    main()
