"""
Camp Yeti autonomous poster.
Reads the persona bible, asks Claude to generate a post in-voice,
generates an image, publishes to Instagram via Blotato, and appends
to the continuity log.

Env vars required:
  ANTHROPIC_API_KEY
  BLOTATO_API_KEY
  BLOTATO_INSTAGRAM_ACCOUNT_ID   (the accountId from Blotato's Accounts page)

Run this on a schedule (see camp_yeti_workflow.yml for GitHub Actions).
"""

import os
import json
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
BLOTATO_API_KEY = os.environ["BLOTATO_API_KEY"]
BLOTATO_ACCOUNT_ID = os.environ["BLOTATO_INSTAGRAM_ACCOUNT_ID"]

PERSONA_PATH = Path(__file__).parent / "camp_yeti_persona_bible.md"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "camp_yeti_agent_system_prompt.md"

ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Blotato's "Image Slideshow" template rendered with a single slide -- this is
# the closest thing Blotato's Create Visual API has to a plain prompt-to-image
# call (most other templates force a text/quote layout onto the image).
# Override via env var if you pick a different template from your Blotato
# dashboard (GET /v2/videos/templates lists the ones available to your account).
BLOTATO_TEMPLATE_ID = os.environ.get(
    "BLOTATO_TEMPLATE_ID", "5903b592-1255-43b4-b9ac-f8ed7cbf6a5f"
)
BLOTATO_VISUAL_POLL_INTERVAL_SECONDS = 4
BLOTATO_VISUAL_POLL_TIMEOUT_SECONDS = 180


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
            "max_tokens": 1500,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(b["text"] for b in data["content"] if b["type"] == "text")


def generate_post(persona_bible: str) -> dict:
    """Ask Claude for a JSON-structured post: caption + image_prompt + log_line."""
    system = (
        "You are the autonomous content generator for the Camp Yeti Instagram "
        "persona. Follow the persona bible exactly. Respond with ONLY valid JSON, "
        "no markdown fences, no preamble, matching this schema:\n"
        '{"caption": "...", "image_prompt": "...", "pillar_used": "...", '
        '"phrase_used": "... or null", "new_lore": "... or null"}'
    )
    user = (
        f"PERSONA BIBLE:\n{persona_bible}\n\n"
        "Generate today's post. Pick a pillar not used in the last 3 log entries. "
        "image_prompt should be a full prompt for an AI image generator, matching "
        "the Visual Identity section precisely (yeti-first silhouette, pink bow, "
        "white/teal fur, on-brand color palette, one hero prop max)."
    )
    raw = call_claude(system, user)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def generate_image_url(image_prompt: str) -> str:
    """
    Generates an image via Blotato's Create Visual API (POST /v2/videos/from-templates)
    and polls until it's rendered, returning the public image URL Blotato hosts it at.

    Uses the "Image Slideshow" template with a single slide and no text overlay --
    Blotato's templates are layout-based (quote cards, carousels, etc.), and this is
    the one that lets `imageSource` be a raw AI prompt with no forced text/caption
    baked into the image itself.
    """
    create_resp = requests.post(
        "https://backend.blotato.com/v2/videos/from-templates",
        headers={
            "blotato-api-key": BLOTATO_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "templateId": BLOTATO_TEMPLATE_ID,
            "inputs": {
                "slides": [{"imageSource": image_prompt, "textOverlay": ""}],
                "aspectRatio": "4:5",
            },
            "render": True,
        },
        timeout=30,
    )
    create_resp.raise_for_status()
    creation_id = create_resp.json()["item"]["id"]

    deadline = time.monotonic() + BLOTATO_VISUAL_POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status_resp = requests.get(
            f"https://backend.blotato.com/v2/videos/creations/{creation_id}",
            headers={"blotato-api-key": BLOTATO_API_KEY},
            timeout=30,
        )
        status_resp.raise_for_status()
        item = status_resp.json()["item"]
        status = item["status"]

        if status == "done":
            image_urls = item.get("imageUrls") or []
            if image_urls:
                return image_urls[0]
            if item.get("mediaUrl"):
                return item["mediaUrl"]
            raise RuntimeError(f"Blotato creation {creation_id} finished with no media URL")

        if status == "creation-from-template-failed":
            raise RuntimeError(f"Blotato visual creation {creation_id} failed: {item}")

        time.sleep(BLOTATO_VISUAL_POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Blotato visual creation {creation_id} did not finish within "
        f"{BLOTATO_VISUAL_POLL_TIMEOUT_SECONDS}s"
    )


def publish_to_instagram(caption: str, image_url: str) -> dict:
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
                    "mediaUrls": [image_url],
                    "platform": "instagram",
                },
                "target": {"targetType": "instagram"},
            }
        },
        timeout=30,
    )
    resp.raise_for_status()
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
        image_url = generate_image_url(post["image_prompt"])
    except Exception as e:
        notify_owner(f"Image generation failed: {e}")
        return

    try:
        result = publish_to_instagram(post["caption"], image_url)
    except requests.HTTPError as e:
        notify_owner(f"Blotato publish failed: {e} -- {e.response.text if e.response else ''}")
        return

    append_log(post)
    print(f"Published: {result}")


if __name__ == "__main__":
    main()
