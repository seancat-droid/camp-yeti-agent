"""
Camp Yeti DM replier.
Polls Blotato for new Instagram DMs, drafts an in-voice reply via Claude (or
skips/escalates per the persona bible's rules), and sends the reply. Tracks
which messages have already been handled in a local state file so it never
double-replies. Mirrors camp_yeti_replier.py's comment-handling pattern.

Note: Blotato's DM read schema couldn't be confirmed against live data (no
DMs had arrived yet when this was written) -- _extract_sender_id() tries the
field names most consistent with the rest of Blotato's API and raises
clearly if none match, so a schema mismatch shows up in the run logs instead
of silently misbehaving.

Env vars required:
  ANTHROPIC_API_KEY
  BLOTATO_API_KEY
  BLOTATO_INSTAGRAM_ACCOUNT_ID

Run this on a schedule (see .github/workflows/camp_yeti_reply.yml).
"""

import os
import json
import requests
from pathlib import Path

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
BLOTATO_API_KEY = os.environ["BLOTATO_API_KEY"]
BLOTATO_ACCOUNT_ID = os.environ["BLOTATO_INSTAGRAM_ACCOUNT_ID"]
PLATFORM = "instagram"

PERSONA_PATH = Path(__file__).parent / "camp_yeti_persona_bible.md"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "camp_yeti_agent_system_prompt.md"
HANDLED_DMS_PATH = Path(__file__).parent / "handled_dms.json"

ANTHROPIC_MODEL = "claude-sonnet-4-6"


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
    if HANDLED_DMS_PATH.exists():
        return set(json.loads(HANDLED_DMS_PATH.read_text()))
    return set()


def save_handled_ids(ids: set):
    HANDLED_DMS_PATH.write_text(json.dumps(sorted(ids), indent=2))


def fetch_new_messages() -> list:
    """Fetches recent Instagram DMs, excluding our own prior sent messages."""
    resp = requests.get(
        "https://backend.blotato.com/v2/messages",
        headers={"blotato-api-key": BLOTATO_API_KEY},
        params={"platform": PLATFORM, "accountId": BLOTATO_ACCOUNT_ID, "limit": 100},
        timeout=30,
    )
    _raise_with_body(resp)
    items = resp.json().get("items", [])
    return [m for m in items if not m.get("isAuthor")]


def _extract_sender_id(message: dict) -> str:
    for key in ("senderId", "recipientId", "userId", "fromId", "authorId"):
        if message.get(key):
            return message[key]
    raise KeyError(f"No recognizable sender-id field on DM: {message}")


def draft_reply(persona_bible: str, system_prompt: str, message_text: str) -> dict:
    """Asks Claude whether/how to reply. Returns
    {"action": "reply"|"skip", "reply": "..." or None, "reason": "..."}."""
    system = (
        "You are Camp Yeti's DM-reply agent. Follow the attached persona "
        "bible and the DM REPLIES + ESCALATION sections of the attached "
        "system prompt exactly. Respond with ONLY valid JSON, no markdown "
        "fences:\n"
        '{"action": "reply" or "skip", "reply": "in-voice reply text or null", '
        '"reason": "brief reason, especially if skipping"}'
    )
    user = (
        f"PERSONA BIBLE:\n{persona_bible}\n\n"
        f"SYSTEM PROMPT:\n{system_prompt}\n\n"
        f'A user sent Camp Yeti a DM: "{message_text}"\n\n'
        "Decide whether to reply in-voice (short, under 2 lines usually) or "
        "skip -- per the escalation rules, skip rather than improvise on "
        "anything hostile/explicit, a real logistical question (e.g. \"is "
        "this a real camp my kid can attend\"), anything involving a minor's "
        "account/image/claim, or anything that reads like it wants a real "
        "personal conversation/relationship with Yeti rather than a bit."
    )
    raw = call_claude(system, user)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def send_message(recipient_id: str, text: str) -> dict:
    resp = requests.post(
        "https://backend.blotato.com/v2/messages",
        headers={
            "blotato-api-key": BLOTATO_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "accountId": BLOTATO_ACCOUNT_ID,
            "recipientId": recipient_id,
            "text": text,
            "target": {"targetType": PLATFORM},
        },
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

    messages = fetch_new_messages()
    new_messages = [m for m in messages if m.get("id") not in handled_ids]

    for message in new_messages:
        msg_id = message.get("id")
        try:
            sender_id = _extract_sender_id(message)
        except KeyError as e:
            notify_owner(f"Skipped DM (unrecognized schema): {e}")
            if msg_id:
                handled_ids.add(msg_id)
            continue

        try:
            decision = draft_reply(persona_bible, system_prompt, message.get("text", ""))
        except Exception as e:
            notify_owner(f"Failed to draft DM reply for {msg_id}: {e}")
            continue

        if decision["action"] == "skip":
            notify_owner(f"Skipped DM {msg_id}: {decision.get('reason')}")
            handled_ids.add(msg_id)
            continue

        try:
            send_message(sender_id, decision["reply"])
            print(f"Replied to DM {msg_id}")
        except Exception as e:
            notify_owner(f"Failed to send DM reply to {msg_id}: {e}")
            continue

        handled_ids.add(msg_id)

    save_handled_ids(handled_ids)


if __name__ == "__main__":
    main()
