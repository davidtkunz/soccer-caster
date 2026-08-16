# soccer-caster

Computer-vision pipeline that turns broadcast soccer footage into a post-game
summary — and then into a live commentary feed running a few seconds behind play.

Built on [supervision](https://github.com/roboflow/supervision) and
[roboflow/sports](https://github.com/roboflow/sports).

## The idea

A live AI commentator has to talk about a play whose outcome it doesn't yet know.
That's what makes it hard, and it's what causes the worst failure mode: confidently
announcing a goal that didn't happen.

So don't run live. Hold a rolling **8-second buffer** of frames and derived state.
Perception runs at the live edge; narration runs eight seconds behind it. By the time
the caster describes a shot being struck, the system has already seen whether it went in.

Broadcasts already run on delay. Nobody notices the eight seconds, and the whole
class of hallucinated-event bugs disappears.

## Status

Early. Phase 1 (post-game summary) is the current target — it shares the entire
perception → state → events backbone with the live caster, minus the latency
pressure and minus the cost of being wrong in public.

| Stage | State |
|---|---|
| 1.1 Shot / replay segmentation | **Implemented** — cut detection, scoreboard localisation, shot classification |
| 1.4 Game state | **Implemented** — possession hysteresis, scoreboard reading, temporal smoothing |
| 1.2, 1.3, 1.5–1.7 | Skeletons with design notes |
| Phase 2 | Skeletons with design notes |

55 tests passing against synthetic footage. **Not yet validated on real
broadcast video** — thresholds will need tuning on actual frames.

### Two rules that shape the state layer

**Hysteresis everywhere.** Raw per-frame readings are noisy in ways that
matter. Nearest-player possession flickers dozens of times a minute between two
players contesting the ball, and every flicker reads as a pass downstream. A
single digit misread reads as a goal. Both are fixed the same way: a new value
must hold for several consecutive frames, and beat the incumbent by a margin,
before it is accepted.

**Constraints reject nonsense for free.** A score never decreases and never
jumps by two. A clock never runs backwards. Applying those rules costs nothing
and discards most misreads before they reach the event layer.

## Phases

**Phase 1 — post-game summary**

| Stage | Module | What it does |
|---|---|---|
| 1.1 | `segmentation.py` | Shot-boundary and replay detection. Must come first — replays otherwise register as duplicate goals. |
| 1.2 | `perception.py` | Player + ball detection, tracking, team assignment. |
| 1.3 | `pitch.py` | Pitch keypoints → homography → canonical 105×68 m coordinates. |
| 1.4 | `state.py` | Per-frame game state; scoreboard OCR for score and clock. |
| 1.5 | `events.py` | Rules over state transitions → discrete events. |
| 1.6 | `eval/` | Hand-labelled evaluation set. Not optional. |
| 1.7 | `summary.py` | Stats, timeline, heatmaps, narrative recap. |

**Phase 2 — live caster**

| Module | What it does |
|---|---|
| `buffer.py` | Ring buffer holding N seconds of frames + derived state. |
| `director.py` | Decides *when* to speak. Importance gate, silence tracking, barge-in. |
| `narration.py` | Streaming text generation. |
| `voice.py` | TTS + pre-rendered interjection playback. |

## Why segmentation comes first

Broadcast feeds cut to close-ups, crowd shots, and replays. Run the event pipeline
on all of it and a replayed goal becomes a second goal.

Two heuristics do most of the work:

- The main tactical view is the one where **pitch keypoints are detectable**.
- The **scoreboard overlay usually disappears during replays**.

Everything downstream runs only on segments classified live-wide.

## Event schema

| Event | Derivation | Confidence |
|---|---|---|
| `pass` | Possession moves between players, same team | High |
| `turnover` | Possession changes team | High |
| `shot` | Ball velocity toward goal mouth, above threshold, from attacking third | Medium |
| `goal` | **Scoreboard delta** (primary); goal-line crossing refines timestamp | High |
| `save` | Shot + keeper contact + no scoreboard change | Medium |
| `restart` | Ball leaves bounds; re-entry point classifies corner / throw / goal kick | High |
| `foul` | Play-stoppage proxy — ball static, players clustered | Low |

Goals are near-certain because the scoreboard tells us. Fouls are genuinely weak
from vision alone — that row is a stub, and no commentary should depend on it.

## Known ceilings

- **Ball tracking caps everything.** Nearly every narratable event revolves around
  the ball, and it's small, fast, motion-blurred and frequently occluded. If ball
  tracking runs at 70%, event detection caps at 70% and commentary quality caps
  below that. Measure it before anything else.
- **Player identity** needs jersey-number OCR, which works on a fraction of frames.
  Resolve by voting across a track.
- **Precision over recall, everywhere.** A caster that says less and is always right
  beats the reverse by a wide margin.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`roboflow/sports` installs from source:

```bash
pip install git+https://github.com/roboflow/sports.git
```

## Tests

```bash
pip install pytest
pytest
```

The segmentation tests run against **synthetic frames** generated in
`tests/synth.py` — green-dominant pitch frames with and without a static score
bug, busy crowd frames, close-ups, and flat graphics. No footage needed, and
each generator targets the property the classifier actually keys on rather than
the appearance of the real thing.

Run segmentation over a real video:

```bash
python -m caster.segmentation match.mp4 --fps 25
```

It prints a shot list with green ratio and scoreboard score per segment, then a
coverage summary. Live wide should be roughly 55–75% of a full match; a much
lower figure usually means calibration locked onto the wrong region rather than
that the broadcaster is unusual.

## Layout

```
src/caster/       pipeline modules
configs/          voice packs, pitch model, thresholds
eval/             hand-labelled ground truth
notebooks/        exploration
```

## License

MIT
