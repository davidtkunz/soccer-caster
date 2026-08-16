"""Narration — turns the event stream into spoken lines.

Two paths with different latency profiles:

  play-by-play   short, reactive, latency-critical. Thinking off, low effort,
                 fast mode, streamed straight into TTS.
  colour         runs during lulls. Adaptive thinking on, medium effort. It has
                 fifteen seconds of dead air to work with.

Prompt layout is fixed by the caching rules. Caching is a prefix match, so the
stable half (persona, commentary rules, match briefing pack) goes first behind a
1-hour cache breakpoint, and everything volatile goes after it. A 90-minute match
plus half-time outruns the 5-minute default TTL, which is why the breakpoint uses
`ttl: "1h"` — the doubled write cost pays for itself within three requests.

Live state (score, clock, substitutions) is injected as a mid-conversation
system message rather than by editing the top-level system field. Editing the
system field would invalidate the entire cached prefix on every score change.
"""

import anthropic

MODEL = "claude-opus-5"
FAST_MODE_BETA = "fast-mode-2026-02-01"

# Written generically on purpose. Naming thinking tags specifically is
# measurably less effective at suppressing them, and any instruction telling the
# model not to reason makes leakage worse rather than better.
OUTPUT_HYGIENE = "Do not include internal or system XML tags in your response."

client = anthropic.Anthropic()


def system_blocks(persona: str, rules: str, briefing_pack: str):
    """Stable prefix. Must be byte-identical across every request of the match.

    Any timestamp, UUID, or rebuilt JSON in here silently invalidates the cache
    and you pay full input price on every line. Verify with
    `usage.cache_read_input_tokens` — if it stays at zero, something in this
    prefix is changing.
    """
    return [
        {"type": "text", "text": f"{persona}\n\n{rules}\n\n{OUTPUT_HYGIENE}"},
        {
            "type": "text",
            "text": briefing_pack,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    ]


def play_by_play(system, recent_turns, event_window, live_state, tts):
    """Stream a short reactive line straight into the voice layer.

    max_tokens is deliberately tight. Lines should be five to fifteen words —
    long sentences add latency and cannot be interrupted cleanly when a goal
    needs to preempt them.
    """
    messages = [
        *recent_turns,
        {"role": "user", "content": event_window},
        # Mid-conversation system message: carries operator authority and sits
        # after the cached history, so the prefix survives the update.
        {"role": "system", "content": live_state},
    ]

    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=150,
        speed="fast",
        betas=[FAST_MODE_BETA],
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
        system=system,
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            tts.feed(chunk)
        return stream.get_final_message()


def colour(system, recent_turns, briefing_selection, live_state, tts):
    """Filler during lulls. Not latency-critical, so leave thinking on."""
    messages = [
        *recent_turns,
        {"role": "user", "content": briefing_selection},
        {"role": "system", "content": live_state},
    ]

    with client.messages.stream(
        model=MODEL,
        max_tokens=400,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=system,
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            tts.feed(chunk)
        return stream.get_final_message()
