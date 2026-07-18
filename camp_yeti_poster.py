"""
Camp Yeti autonomous poster.
Reads the persona bible, asks Claude to generate a post in-voice,
generates a voiced video, publishes to Instagram via Blotato, and appends
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

# Blotato's "AI Video with AI Voice" template -- cheaper in credits than the
# consistent-character template (which ran out of credits mid-render), at the
# cost of a fresh AI reinterpretation of the character each scene instead of
# true visual consistency. Override via env var if you pick a different
# template (GET /v2/videos/templates lists what's available to your account).
BLOTATO_TEMPLATE_ID = os.environ.get(
    "BLOTATO_TEMPLATE_ID", "/base/v2/ai-story-video/5903fe43-514d-40ee-a060-0d6628c5f8fd/v1"
)
# Must match one of Blotato's exact voiceName strings, descriptor included.
BLOTATO_VOICE_NAME = os.environ.get("BLOTATO_VOICE_NAME", "Callum (Transatlantic, intense)")
BLOTATO_VISUAL_POLL_INTERVAL_SECONDS = 5
# Video-with-voiceover renders take noticeably longer than a single image.
BLOTATO_VISUAL_POLL_TIMEOUT_SECONDS = 420
# Optional: name of a track from Instagram's own licensed audio library to
# attach to the reel (Blotato's audioName field). Untested/undocumented on
# Blotato's side -- unset by default; set this once you've confirmed a value
# actually attaches audio rather than being silently ignored.
BLOTATO_AUDIO_NAME = os.environ.get("BLOTATO_AUDIO_NAME")


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
            "max_tokens": 1500,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=60,
    )
    _raise_with_body(resp)
    data = resp.json()
    return "".join(b["text"] for b in data["content"] if b["type"] == "text")


def generate_post(persona_bible: str) -> dict:
    """Ask Claude for a JSON-structured post: caption + scene-by-scene video script."""
    system = (
        "You are the autonomous content generator for the Camp Yeti Instagram "
        "persona. Follow the persona bible exactly. Respond with ONLY valid JSON, "
        "no markdown fences, no preamble, matching this schema:\n"
        '{"caption": "...", "scenes": [{"visual": "...", "script": "..."}, ...], '
        '"pillar_used": "...", "phrase_used": "... or null", "new_lore": "... or null"}'
    )
    user = (
        f"PERSONA BIBLE:\n{persona_bible}\n\n"
        "Generate today's post as a short voiced video, 2-4 scenes. Pick a pillar "
        "not used in the last 3 log entries.\n\n"
        "Each scene needs:\n"
        "- visual: describe the SETTING and what Yeti is doing/reacting to in "
        "this scene (location, action, one hero prop max if any). Her physical "
        "appearance is locked to a reference image, so don't re-describe her "
        "body, fur, or face here -- only what's happening around and to her.\n"
        "- script: one line Yeti actually SAYS ALOUD in that scene, in her voice "
        "per the Voice Rules section -- insult-comedy energy, a cutting deadpan "
        "'read' rather than gentle whimsy, delivered dry with no softening wink. "
        "Short declaratives, third-person threats, the mock-sermon-undercut-by-"
        "one-blunt-line structure works well across scenes -- e.g. two solemn "
        "scenes building the 'teaching,' then a final scene with the unhinged "
        "gag line.\n\n"
        "caption is the separate Instagram post caption (not read aloud) -- 1-4 "
        "lines, #CAMPYETI plus at most one or two theme tags."
    )
    raw = call_claude(system, user)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def generate_video_url(scenes: list) -> str:
    """
    Generates a voiced video via Blotato's Create Visual API
    (POST /v2/videos/from-templates, "AI Video with AI Voice" template) and
    polls until it's rendered, returning the public video URL Blotato hosts
    it at. Each scene's AI-generated visual is narrated by an ElevenLabs
    voice reading that scene's script line.
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
                "scenes": [
                    {"mediaSource": scene["visual"], "script": scene["script"]}
                    for scene in scenes
                ],
                "voiceName": BLOTATO_VOICE_NAME,
                "aspectRatio": "9:16",
                "captionPosition": "bottom",
            },
            "render": True,
        },
        timeout=30,
    )
    _raise_with_body(create_resp)
    creation_id = create_resp.json()["item"]["id"]

    deadline = time.monotonic() + BLOTATO_VISUAL_POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status_resp = requests.get(
            f"https://backend.blotato.com/v2/videos/creations/{creation_id}",
            headers={"blotato-api-key": BLOTATO_API_KEY},
            timeout=30,
        )
        _raise_with_body(status_resp)
        item = status_resp.json()["item"]
        status = item["status"]

        if status == "done":
            if item.get("mediaUrl"):
                return item["mediaUrl"]
            image_urls = item.get("imageUrls") or []
            if image_urls:
                return image_urls[0]
            raise RuntimeError(f"Blotato creation {creation_id} finished with no media URL")

        if status in ("creation-from-template-failed", "insufficient-credits"):
            raise RuntimeError(
                f"Blotato visual creation {creation_id} failed: "
                f"{item.get('error') or item}"
            )

        time.sleep(BLOTATO_VISUAL_POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Blotato visual creation {creation_id} did not finish within "
        f"{BLOTATO_VISUAL_POLL_TIMEOUT_SECONDS}s"
    )


def publish_to_instagram(caption: str, video_url: str) -> dict:
    target = {"targetType": "instagram", "mediaType": "reel"}
    if BLOTATO_AUDIO_NAME:
        target["audioName"] = BLOTATO_AUDIO_NAME

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
                "target": target,
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
        video_url = generate_video_url(post["scenes"])
    except Exception as e:
        notify_owner(f"Video generation failed: {e}")
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
