# Camp Yeti weekly improvement log

Append-only. Each entry: date, area worked (rotates 1→2→3→1...), what
changed/was found, what still needs the human owner's call, and what the
next run should pick up.

---

## 2026-11-09 -- Area 1: Video quality

First run of this rotation (no prior log existed) -- started at area 1.

**What changed:**
Added `ai_video_backend.py`: a prototype client for Runway's Gen-4 Turbo
image-to-video API, plus a thin opt-in hook in `camp_yeti_poster.py`
(`build_video()` / `CAMP_YETI_AI_VIDEO_PROVIDER` env var, replacing the
direct `build_animated_video()` call in `main()`). With the env var unset
(the current state everywhere, including the Actions workflow), behavior is
byte-for-byte identical to before -- `build_video()` just calls
`build_animated_video()`. Nothing was activated; no account was created, no
payment info entered, no secret added anywhere.

**Why this instead of the current pipeline:** the local
bounce/blink/zoom/Ken-Burns animation in `build_animated_video()` is real
rendered video, but it's always the same flat character cutout bouncing in
place on a static background -- no camera-aware motion, no secondary
animation (fur/fabric sway, hand gestures), no depth. An image-to-video
model fed the exact composited frame (`render_text_card_image()` -- same
character art, same text card, same background) can add that motion while
keeping the identical reference artwork, so nothing about her design
changes.

**Options researched (Nov 2026 pricing, sourced from third-party pricing
blogs that disagree with each other -- confirm on each vendor's own pricing
page before paying anything):**

| Provider | ~Cost/sec | Notes |
|---|---|---|
| MiniMax Hailuo 02 | $0.017-0.045 | Cheapest found; no independent read on character-consistency quality |
| ByteDance Seedance 2.0 Fast | $0.022-0.09 | Range depends on direct vs. reseller route |
| Google Veo 3.1 Lite | ~$0.03 (720p, video-only) | First-party list price lowest found for Veo family |
| **Runway Gen-4 Turbo** | **~$0.05** | **Prototyped** -- most stable/documented API of this group; image+text-prompt -> video, single-clip cap of 10s |
| Kling 3.0 / O1 | $0.08-0.17 | Vendor claims strongest character consistency; priced noticeably higher |

A 16s reel (this account's default `TARGET_REEL_DURATION_SECONDS`) at Gen-4
Turbo's rate is roughly $0.50-$0.80/post at daily-ish cadence; the account
posts on a 2-3 day alternating gap, so call it ~$0.20-0.30/day average, or
very roughly **$6-10/month** if run on every post. That's napkin math from
third-party-sourced per-second rates, not a vendor quote -- re-check before
budgeting for real.

**What still needs the human owner's decision (nothing auto-activates):**
1. Whether this is worth $6-10+/month at all vs. the current free pipeline.
2. If yes: sign up for a Runway API account, get an API key, and set it as
   `RUNWAY_API_KEY` *and* `CAMP_YETI_AI_VIDEO_PROVIDER=runway` as repo
   secrets/env vars in the Actions workflow (`.github/workflows/camp_yeti_post.yml`'s
   env block) -- this agent will never do that step itself per the hard
   rule against signing up for paid services.
3. Before flipping it on for real: verify `ai_video_backend.py`'s Runway
   request/response field names (`promptImage`, `promptText`, `ratio`,
   `duration`, polling via `/v1/tasks/{id}`) against Runway's *current* docs
   at docs.dev.runwayml.com -- this environment's network couldn't reach
   that host to confirm them directly, so they're researched-but-unverified.
4. Gen-4 Turbo caps a single generation at 10s, short of the account's
   16-30s reel length -- the prototype generates one clip and leaves it at
   that length rather than chaining multiple generations into a longer
   reel. Worth deciding whether a shorter AI clip is an acceptable tradeoff,
   or whether clip-chaining needs to be built first.
5. Output aspect ratio: Gen-4 Turbo doesn't offer an exact 4:5 (1080x1350)
   option, so the prototype requests the closest supported ratio (832:1104)
   and pads back to 4:5 in the ffmpeg mux step -- worth an eyeball check on
   a real generated clip for letterboxing before shipping.

**Next run should pick up:** Area 2 (Engagement) -- improvements to
`camp_yeti_replier.py` / `camp_yeti_dm_replier.py` or posting-caption
variety. After that, Area 3 (Growth strategy notes). Then back to Area 1 --
if the owner has made a call on the above by then, the next video-quality
run could prototype a second provider (Kling, for the character-consistency
claim) for comparison, or build the multi-clip chaining needed to hit full
reel length.
