"""
Camp Yeti DM replier.
Polls the Instagram Graph API's Conversations edge for new Instagram DMs,
drafts an in-voice reply via Claude (or skips/escalates per the persona
bible's rules), and sends the reply. Tracks which messages have already been
handled in a local state file so it never double-replies. Mirrors
camp_yeti_replier.py's comment-handling pattern.

Env vars required:
  ANTHROPIC_API_KEY
  META_PAGE_ACCESS_TOKEN   (needs instagram_manage_messages, instagram_basic,
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
HANDLED_DMS_PATH = Path(__file__).parent / "handled_dms.json"

ANTHROPIC_MODEL = "claude-sonnet-4-6"

RECENT_CONVERSATIONS_LIMIT = 50
MESSAGES_PER_CONVERSATION_LIMIT = 20  # only the tail of each conversation needs checking


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
    """Fetches the most recent message in each of Camp Yeti's Instagram DM
    conversations, excluding messages we sent ourselves (from.id ==
    META_IG_BUSINESS_ID). Each message dict is annotated with sender_id so a
    reply can be sent to the right person."""
    conversations_resp = requests.get(
        f"{GRAPH_BASE}/{META_IG_BUSINESS_ID}/conversations",
        params={
            "platform": "instagram",
            "limit": RECENT_CONVERSATIONS_LIMIT,
            "access_token": META_PAGE_ACCESS_TOKEN,
        },
        timeout=30,
    )
    _raise_with_body(conversations_resp)
    conversations = conversations_resp.json().get("data", [])

    messages = []
    for convo in conversations:
        messages_resp = requests.get(
            f"{GRAPH_BASE}/{convo['id']}",
            params={
                "fields": f"messages.limit({MESSAGES_PER_CONVERSATION_LIMIT}){{id,from,message}}",
                "access_token": META_PAGE_ACCESS_TOKEN,
            },
            timeout=30,
        )
        _raise_with_body(messages_resp)
        for msg in messages_resp.json().get("messages", {}).get("data", []):
            sender_id = msg.get("from", {}).get("id")
            if sender_id and sender_id != META_IG_BUSINESS_ID:
                messages.append({
                    "id": msg["id"],
                    "text": msg.get("message", ""),
                    "sender_id": sender_id,
                })
    return messages


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
        f"{GRAPH_BASE}/{META_IG_BUSINESS_ID}/messages",
        params={"access_token": META_PAGE_ACCESS_TOKEN},
        json={"recipient": {"id": recipient_id}, "message": {"text": text}},
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
    new_messages = [m for m in messages if m["id"] not in handled_ids]

    for message in new_messages:
        msg_id = message["id"]
        try:
            decision = draft_reply(persona_bible, system_prompt, message["text"])
        except Exception as e:
            notify_owner(f"Failed to draft DM reply for {msg_id}: {e}")
            continue

        if decision["action"] == "skip":
            notify_owner(f"Skipped DM {msg_id}: {decision.get('reason')}")
            handled_ids.add(msg_id)
            continue

        try:
            send_message(message["sender_id"], decision["reply"])
            print(f"Replied to DM {msg_id}")
        except Exception as e:
            notify_owner(f"Failed to send DM reply to {msg_id}: {e}")
            continue

        handled_ids.add(msg_id)

    save_handled_ids(handled_ids)


if __name__ == "__main__":
    main()
