"""
Camp Yeti comment replier.
Polls the Instagram Graph API for new comments on Camp Yeti's recent posts,
drafts an in-voice reply via Claude (or skips/escalates per the persona
bible's rules), and posts the reply. Tracks which comments have already been
handled in a local state file so it never double-replies.

Only top-level comments are fetched from each post's /comments edge -- our
own replies live under each comment's own /replies edge instead, so they
never show up here and don't need separate filtering.

Env vars required:
  ANTHROPIC_API_KEY
  META_PAGE_ACCESS_TOKEN   (needs instagram_manage_comments, instagram_basic,
                             pages_read_engagement, pages_show_list)
  META_IG_BUSINESS_ID

Run this on a schedule (see .github/workflows/camp_yeti_reply.yml).
"""

import os
import json
import requests
from pathlib import Path

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
META_PAGE_ACCESS_TOKEN = os.environ["META_PAGE_ACCESS_TOKEN"]
META_IG_BUSINESS_ID = os.environ["META_IG_BUSINESS_ID"]
META_GRAPH_VERSION = "v20.0"
GRAPH_BASE = f"https://graph.facebook.com/{META_GRAPH_VERSION}"

PERSONA_PATH = Path(__file__).parent / "camp_yeti_persona_bible.md"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "camp_yeti_agent_system_prompt.md"
HANDLED_COMMENTS_PATH = Path(__file__).parent / "handled_comments.json"

ANTHROPIC_MODEL = "claude-sonnet-4-6"

RECENT_MEDIA_LIMIT = 25  # how many of the most recent posts to check for new comments
COMMENTS_PER_MEDIA_LIMIT = 50


def _raise_with_body(resp: requests.Response):
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
            "max_tokens": 500,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=60,
    )
    _raise_with_body(resp)
    data = resp.json()
    return "".join(b["text"] for b in data["content"] if b["type"] == "text")


def load_handled_ids() -> set:
    if HANDLED_COMMENTS_PATH.exists():
        return set(json.loads(HANDLED_COMMENTS_PATH.read_text()))
    return set()


def save_handled_ids(ids: set):
    HANDLED_COMMENTS_PATH.write_text(json.dumps(sorted(ids), indent=2))


def fetch_new_comments() -> list:
    """Fetches top-level comments on Camp Yeti's most recent posts. Each
    comment dict is annotated with media_id so a reply can be posted back to
    the right thread."""
    media_resp = requests.get(
        f"{GRAPH_BASE}/{META_IG_BUSINESS_ID}/media",
        params={
            "fields": "id",
            "limit": RECENT_MEDIA_LIMIT,
            "access_token": META_PAGE_ACCESS_TOKEN,
        },
        timeout=30,
    )
    _raise_with_body(media_resp)
    media_items = media_resp.json().get("data", [])

    comments = []
    for media in media_items:
        comments_resp = requests.get(
            f"{GRAPH_BASE}/{media['id']}/comments",
            params={
                "fields": "id,text,username",
                "limit": COMMENTS_PER_MEDIA_LIMIT,
                "access_token": META_PAGE_ACCESS_TOKEN,
            },
            timeout=30,
        )
        _raise_with_body(comments_resp)
        for comment in comments_resp.json().get("data", []):
            comment["media_id"] = media["id"]
            comments.append(comment)
    return comments


def draft_reply(persona_bible: str, system_prompt: str, comment_text: str) -> dict:
    """Asks Claude whether/how to reply. Returns
    {"action": "reply"|"skip", "reply": "..." or None, "reason": "..."}."""
    system = (
        "You are Camp Yeti's comment-reply agent. Follow the attached persona "
        "bible and the COMMENT/DM REPLIES + ESCALATION sections of the attached "
        "system prompt exactly. Respond with ONLY valid JSON, no markdown "
        "fences:\n"
        '{"action": "reply" or "skip", "reply": "in-voice reply text or null", '
        '"reason": "brief reason, especially if skipping"}'
    )
    user = (
        f"PERSONA BIBLE:\n{persona_bible}\n\n"
        f"SYSTEM PROMPT:\n{system_prompt}\n\n"
        f'A user commented on a Camp Yeti Instagram post: "{comment_text}"\n\n'
        "Decide whether to reply in-voice (short, under 2 lines usually) or "
        "skip -- per the escalation rules, skip rather than improvise on "
        "anything hostile/explicit, a real logistical question (e.g. \"is "
        "this a real camp my kid can attend\"), or anything involving a "
        "minor's account, image, or claim."
    )
    raw = call_claude(system, user)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def post_reply(comment_id: str, text: str) -> dict:
    resp = requests.post(
        f"{GRAPH_BASE}/{comment_id}/replies",
        params={"message": text, "access_token": META_PAGE_ACCESS_TOKEN},
        timeout=30,
    )
    _raise_with_body(resp)
    return resp.json()


def notify_owner(message: str):
    """Wire this to email/Slack/SMS -- whatever reaches you. Placeholder: prints."""
    print(f"[ESCALATION] {message}")


def main():
    persona_bible = PERSONA_PATH.read_text()
    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    handled_ids = load_handled_ids()

    comments = fetch_new_comments()
    new_comments = [c for c in comments if c["id"] not in handled_ids]

    for comment in new_comments:
        try:
            decision = draft_reply(persona_bible, system_prompt, comment["text"])
        except Exception as e:
            notify_owner(f"Failed to draft reply for comment {comment['id']}: {e}")
            continue

        if decision["action"] == "skip":
            notify_owner(f"Skipped comment {comment['id']}: {decision.get('reason')}")
            handled_ids.add(comment["id"])
            continue

        try:
            post_reply(comment["id"], decision["reply"])
            print(f"Replied to comment {comment['id']}")
        except Exception as e:
            notify_owner(f"Failed to post reply to comment {comment['id']}: {e}")
            continue

        handled_ids.add(comment["id"])

    save_handled_ids(handled_ids)


if __name__ == "__main__":
    main()
