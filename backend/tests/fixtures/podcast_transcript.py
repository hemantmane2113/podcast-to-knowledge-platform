"""A representative long-form podcast transcript fixture (Phase 3A §3A.12).

Synthetic dev/test data ONLY — this was written by hand for this repo, not
retrieved from Supadata or any real podcast/video. It exists to exercise
the cleaner and chunker against something shaped like real auto-caption
output: short and long segments, natural pauses of varying size, clear
sentence boundaries, several topic transitions, a run of segments with
almost no gap between them, and one deliberately long uninterrupted
section with no internal punctuation (to exercise safe mid-text
splitting).

`SEGMENT_SCRIPT` entries are (text, gap_after_ms) — the gap is the pause
before the *next* segment starts. `build_transcript_segments()` turns
these into TranscriptSegment ORM objects with computed start_ms/duration_ms/
sequence_number, ready to hand to the cleaner and chunker exactly like a
real ingested transcript would be.
"""

import uuid

from app.models.transcript_segment import TranscriptSegment

_WORDS_PER_SECOND = 2.5  # ~150 wpm, a typical conversational pace
_MIN_DURATION_MS = 700

# --- Section 1: cold open / guest introduction (short segments, small gaps) ----
_INTRO = [
    ("Welcome back to the show.", 250),
    ("Today I'm talking with someone who's spent the last decade building large scale systems.", 300),
    ("Thanks for having me, it's great to be here.", 350),
    ("So let's just start at the beginning.", 200),
    ("How did you first get into this field?", 900),  # larger pause before guest answers
    (
        "Honestly, it was kind of an accident. I was working on something completely "
        "unrelated, a distributed database project, and I kept running into the same "
        "wall over and over again.",
        400,
    ),
    ("What was the wall?", 300),
    ("Coordination. Getting a bunch of machines to agree on anything at scale is brutal.", 1600),
]

# --- Section 2: scaling laws (longer segments, clear sentence boundaries) -----
_SCALING = [
    (
        "Let's talk about scaling laws, because I think that's where a lot of the "
        "recent progress has actually come from.",
        250,
    ),
    (
        "Right. The basic idea is that if you plot loss against compute, data, and "
        "parameters on a log-log scale, you get a remarkably smooth curve.",
        300,
    ),
    (
        "And that smoothness is what let people make confident predictions about "
        "systems that hadn't been built yet.",
        250,
    ),
    ("It's not magic, but it sure looked like magic the first time I saw it.", 1800),  # pause
    (
        "A lot of people hear scaling laws and assume it just means bigger is better.",
        200,
    ),
    (
        "That's the part I think gets misunderstood the most. It's not simply about "
        "making the model larger.",
        150,
    ),
    ("The gains come from the interaction between compute, data, and algorithms together.", 250),
    ("If you scale one of those without the others, you hit diminishing returns fast.", 2100),  # topic pause
]

# --- Section 3: rapid back-and-forth, almost no gap between segments ----------
_RAPID_EXCHANGE = [
    ("Data quality matters too, right?", 80),
    ("Enormously.", 90),
    ("More than people think?", 70),
    ("Much more.", 60),
    ("Even a small amount of bad data can hurt more than a lot of mediocre data helps.", 1400),
]

# --- Section 4: topic transition — agents and tool use ------------------------
_AGENTS = [
    ("So where does that leave agents? Everyone's talking about agents right now.", 300),
    (
        "I think agents are genuinely useful, but the framing is often wrong. "
        "People talk about them like it's one capability, but it's really a "
        "composition of several things working together.",
        350,
    ),
    ("Planning, tool use, memory, and error recovery.", 250),
    (
        "Right, and most of the visible failures come from the error recovery part, "
        "not the planning part.",
        400,
    ),
    ("An agent that can't notice it made a mistake is far more dangerous than a slow one.", 2500),  # big pause, topic shift coming
]

# --- Section 5: one unusually long, uninterrupted section, NO internal
# sentence punctuation -- exercises safe word-boundary splitting when a
# single segment alone exceeds the max chunk size. -----------------------------
_LONG_UNINTERRUPTED = [
    (
        "and honestly if you look back at the last five years of progress and you "
        "try to trace where each meaningful jump actually came from you find that it "
        "is almost never a single clever trick that nobody thought of before it is "
        "much more often a combination of a slightly better training setup slightly "
        "cleaner data a slightly larger run and just enough compute to let the whole "
        "thing actually converge and none of those individually look impressive in "
        "isolation which is part of why it is so easy for outside observers to miss "
        "what is going on and assume there must be some secret architectural insight "
        "driving everything when really it is mostly relentless engineering applied "
        "consistently over a long period of time and that is a much less exciting "
        "story to tell but it happens to be the true one",
        3000,
    ),
]

# --- Section 6: economics / business impact (topic transition) ----------------
_ECONOMICS = [
    ("Let's shift to the economic side of this.", 250),
    ("What's actually changing for companies deploying these systems day to day?", 300),
    (
        "The biggest shift I've seen is in where engineering time goes. Less time "
        "writing boilerplate, more time reviewing and steering output.",
        350,
    ),
    ("Is that a net win?", 200),
    ("For senior engineers, usually yes. For teams without strong review practices, not always.", 2000),
]

# --- Section 7: safety and alignment (topic transition, mixed lengths) --------
_SAFETY = [
    ("I want to bring up safety before we run out of time.", 300),
    (
        "Sure. I'd split that into two very different conversations, honestly, "
        "because people mean different things by the word.",
        250,
    ),
    ("Near-term reliability versus long-term existential risk.", 300),
    (
        "Right. And most of my day-to-day work is entirely in the first bucket. "
        "Making sure a system doesn't confidently state something false, making "
        "sure it doesn't take an irreversible action it shouldn't take.",
        350,
    ),
    ("Those sound like basic engineering problems.", 200),
    ("They are, mostly. They're just unusually hard basic engineering problems.", 1900),
    (
        "The long-term risk conversation is real too, I don't want to dismiss it, "
        "but I think it pulls attention away from the unglamorous work that "
        "actually reduces harm today.",
        2200,
    ),
]

# --- Section 8: hardware and compute economics ---------------------------------
_HARDWARE = [
    ("Let's talk about hardware for a minute, because I think it's underrated.", 250),
    (
        "Compute cost has been dropping, but demand has been growing even faster, "
        "so the net effect for most teams is that budgets are still a real constraint.",
        300,
    ),
    ("Is that mostly GPUs, or is the story more complicated than that?", 200),
    (
        "GPUs dominate the conversation, but a lot of the actual cost in production "
        "systems comes from memory bandwidth and networking between machines, not "
        "raw compute.",
        350,
    ),
    ("That's the part people don't see in the headlines.", 150),
    ("Exactly. Nobody writes an exciting article about interconnect bandwidth.", 2400),
]

# --- Section 9: evaluation and benchmarks ---------------------------------------
_EVALUATION = [
    ("How do you even know if a change made things better?", 250),
    (
        "Honestly, this is the part of the job most people underestimate. Good "
        "evaluation is harder than good modeling, in my experience.",
        300,
    ),
    ("Public benchmarks help, but they saturate, and they get gamed.", 250),
    (
        "We ended up building a lot of our own evaluation sets, specific to the "
        "actual tasks our users care about, rather than trusting a leaderboard "
        "number.",
        400,
    ),
    ("Does that scale, building custom evals for everything?", 200),
    ("Not infinitely, no. But it scales a lot better than shipping blind.", 1700),
]

# --- Section 10: predictions / closing thoughts on the field --------------------
_PREDICTIONS = [
    ("If you had to guess, what changes most in the next couple of years?", 300),
    (
        "I think the boring stuff wins. Better tooling, better evaluation, better "
        "data pipelines. Less flashy than a new architecture, but that's usually "
        "where the real gains hide.",
        350,
    ),
    ("Any predictions you'd be willing to be wrong about publicly?", 300),
    (
        "Sure. I think a lot of the current agent frameworks look completely "
        "different in two years, once people figure out which parts of the "
        "abstraction were actually load-bearing.",
        1800,
    ),
]

# --- Section 11: a real deployment failure story (narrative, longer segments) --
_WAR_STORY = [
    ("Can you tell a story about something that went wrong in production?", 300),
    (
        "Oh, plenty of those. There's one I still think about. We shipped a change "
        "that looked completely safe in every offline evaluation we ran.",
        250,
    ),
    ("What happened?", 200),
    (
        "In production, under real traffic patterns, the system started retrying a "
        "specific kind of failed request far more aggressively than we intended, "
        "and that retry storm took down a downstream service that had nothing to "
        "do with the original change.",
        350,
    ),
    ("None of your evaluations caught that?", 200),
    (
        "None of them, because none of them modeled retry behavior under partial "
        "outages. It just wasn't a scenario anyone had written a test for.",
        1600,
    ),
    ("What changed after that?", 250),
    ("We started testing failure modes explicitly, not just correctness.", 2000),
]

# --- Section 12: open source versus closed ecosystems ---------------------------
_OPEN_SOURCE = [
    ("Where do you land on open source models versus closed ones?", 300),
    (
        "I don't think it's really one debate. For research and reproducibility, "
        "open weights are enormously valuable. For some production use cases, a "
        "hosted closed model is just the pragmatic choice.",
        350,
    ),
    ("Do you think that gap closes over time?", 200),
    ("Mostly, yes. The gap has been shrinking pretty consistently for a couple of years.", 1900),
]

# --- Section 13: team practices and hiring ---------------------------------------
_TEAM_PRACTICES = [
    ("Let's talk about the team side of this for a bit.", 250),
    ("What do you actually look for when you're hiring for this kind of work?", 300),
    (
        "Curiosity about failure cases, more than anything else. The strongest "
        "people I've worked with are the ones who get suspicious when something "
        "works a little too well on the first try.",
        350,
    ),
    ("That's a good filter.", 200),
    ("It's saved us more than once.", 2100),
]

# --- Section 14: closing (short segments, natural wind-down) -------------------
_CLOSING = [
    ("This has been a great conversation.", 200),
    ("Any final thoughts before we wrap up?", 400),
    ("Just that the boring parts of this work are usually the important parts.", 300),
    ("Data, evaluation, and patience.", 250),
    ("Thanks so much for coming on.", 200),
    ("Thanks for having me.", 0),
]

SEGMENT_SCRIPT: list[tuple[str, int]] = (
    _INTRO
    + _SCALING
    + _RAPID_EXCHANGE
    + _AGENTS
    + _LONG_UNINTERRUPTED
    + _ECONOMICS
    + _SAFETY
    + _HARDWARE
    + _EVALUATION
    + _WAR_STORY
    + _OPEN_SOURCE
    + _TEAM_PRACTICES
    + _PREDICTIONS
    + _CLOSING
)


def _estimate_duration_ms(text: str) -> int:
    word_count = len(text.split())
    return max(_MIN_DURATION_MS, round((word_count / _WORDS_PER_SECOND) * 1000))


def build_transcript_segments(transcript_id: uuid.UUID | None = None) -> list[TranscriptSegment]:
    """Builds unpersisted TranscriptSegment ORM objects from SEGMENT_SCRIPT,
    with sequential timing computed the same way a real transcript's
    timing would look (each segment's start_ms follows directly from the
    previous one's end plus its trailing gap).
    """
    tid = transcript_id or uuid.uuid4()
    segments: list[TranscriptSegment] = []
    cursor_ms = 0

    for index, (text, gap_after_ms) in enumerate(SEGMENT_SCRIPT):
        duration_ms = _estimate_duration_ms(text)
        segments.append(
            TranscriptSegment(
                id=uuid.uuid4(),
                transcript_id=tid,
                sequence_number=index,
                text=text,
                start_ms=cursor_ms,
                duration_ms=duration_ms,
            )
        )
        cursor_ms += duration_ms + gap_after_ms

    return segments
