"""Stage 1: cheap, deterministic candidate generation + rule-based scoring.

Every candidate promo is cut on CAMERA shot boundaries (natural edit points) and never
splits a line of dialogue. Rules score four things the brief asks for:
  hook        - something in the first 3 seconds that stops the scroll
  cliffhanger - the clip ends unresolved
  standalone  - a cold viewer can follow it
  intensity   - conflict/emotion density across the whole clip
The rules are deliberately transparent so an editor can see *why* a clip ranked high.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import Episode, Event, fmt

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0
TAIL_WINDOW = 6.0

# --- lexicons (edit these per genre) -------------------------------------------------
PHYSICAL = r"slap|strik|struck|punch|grab|seiz|shov|hurl|throw|threw|collaps|faint|knife|dagger|sword|blade|blood|bleed|fire|flame|blaze|inferno|explo|shatter|smash|crash|chain|chok|kiss|drag|whip|crop|red-hot|iron|scream|gasp|lung|tackl|yank|wrench|stab|fall|kneel|drops to one knee|tears? (?:stream|spill)|cuff|shackle|noose"
REVEAL = r"duke|duchess|heir|pregnan|poison|arsenic|witch|execut|hang|gallows|dead|alive|truth|lie|liar|betray|divorce|marry|married|marriage|wedding|invalid|arrest|treason|guilty|murder|kill|die|stole|thief|theft|son|disown|bastard|husband|wife"
INTENSE_TAGS = r"furious|scream|shock|horrif|enrag|desperate|sob|gloat|menac|threat|venom|outrag|terrif|defian|cold|command|mock|cruel|frantic|panic|roar|shout|livid|seeth|devastat|hiss|snarl|vicious|sneer|accus|fierce|lethal|icy|savage|furi|disbelief|stunned|aghast|sadistic|triumph|smug|contempt|scorn|ruthless|grim|rage"
CALM_TAGS = r"soft|tender|gentle|warm|calm|quiet|relieved|loving|content|peace|fond|playful|amused|wistful|sleepy"
STING_SFX = r"cliffhanger|sting|ominous|boom|thunder|gasp|crash|shatter|slam|explo|scream|silence|drone|swell|thud|crack|slap"
CUT_TO_BLACK = r"cut to black|smash cut|freeze"
RESOLUTION = r"embrac|smil|laugh|relie|thank|hug|safe now|it's over|forgive|peaceful|fades? out|sleep"
THREAT_LINE = r"\bwill\b|never|unless|or else|\bdie\b|kill|arrest|hang|burn|destroy|regret|pay for|you'll|before .* (?:dawn|sunrise)|tomorrow"
BACKREF = r"\bagain\b|\bstill\b|remember|earlier|as i said|that night|after what|last time|like before|once more|you promised"
RELATION = r"husband|wife|sister|mother|father|fianc|duke|duchess|maid|son|daughter|bride|groom|stepmother|family"


def _n(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, re.I))


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class Segment:
    episode: int
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {"episode": self.episode, "start": fmt(self.start), "end": fmt(self.end),
                "start_s": self.start, "end_s": self.end}


@dataclass
class Candidate:
    kind: str                    # "single" | "flash-forward"
    segments: list[Segment]
    events: list[list[Event]]    # events per segment
    scores: dict = field(default_factory=dict)
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    flags: list[str] = field(default_factory=list)
    llm: dict | None = None

    @property
    def duration(self) -> float:
        return sum(s.dur for s in self.segments)

    @property
    def total(self) -> float:
        return self.scores.get("final", self.scores.get("rules", 0.0))

    def transcript(self) -> str:
        """Clip-relative transcript: what a cold viewer would see/hear, nothing else."""
        out, offset_acc = [], 0.0
        for seg, evs in zip(self.segments, self.events):
            if len(self.segments) > 1:
                out.append(f"--- segment: episode {seg.episode} {fmt(seg.start)}-{fmt(seg.end)} ---")
            for e in evs:
                if e.kind == "camera":
                    continue
                out.append(e.line(offset=seg.start - offset_acc))
            offset_acc += seg.dur
        return "\n".join(out)


# --- candidate generation ------------------------------------------------------------

def cut_points(ep: Episode) -> list[float]:
    """Shot boundaries that don't fall inside a line of dialogue."""
    pts = {0.0, ep.duration}
    for e in ep.events:
        if e.kind == "camera":
            pts.add(e.start)
            pts.add(e.end)
    lines = [e for e in ep.events if e.kind == "dialogue"]
    ok = [t for t in pts if not any(d.start < t - 0.15 and d.end > t + 0.15 for d in lines)]
    return sorted(ok)


def events_in(ep: Episode, start: float, end: float) -> list[Event]:
    """Events that start inside [start, end)."""
    return [e for e in ep.events if start - 0.05 <= e.start < end - 0.05]


def single_candidates(ep: Episode) -> list[Candidate]:
    pts = cut_points(ep)
    cands = []
    for i, s in enumerate(pts):
        for t in pts[i + 1:]:
            if t - s < MIN_LEN:
                continue
            if t - s > MAX_LEN:
                break
            seg = Segment(ep.number, s, t)
            cands.append(Candidate("single", [seg], [events_in(ep, s, t)]))
    return cands


# --- rule scoring --------------------------------------------------------------------

def _hook(evs: list[Event], seg_start: float, ep_start_bonus: bool) -> tuple[float, str]:
    first = [e for e in evs if e.start < seg_start + HOOK_WINDOW]
    best, best_txt = 0.0, ""
    for e in first:
        s = 0.0
        txt = e.text
        if e.kind == "dialogue":
            tags = " ".join(e.tags)
            s += 0.35 + 0.25 * min(_n(INTENSE_TAGS, tags), 2) + 0.2 * min(_n(REVEAL, txt), 2)
            s += 0.15 if txt.endswith(("!", "?")) else 0
            s -= 0.3 * min(_n(CALM_TAGS, tags), 1)
            txt = f'{e.speaker}: "{e.text}"'
        elif e.kind in ("action", "camera"):
            s += 0.3 * min(_n(PHYSICAL, txt), 3) + 0.1 * min(_n(REVEAL, txt), 2)
            s *= 1.0 if e.kind == "action" else 0.7
        elif e.kind == "sfx":
            s += 0.25 * min(_n(STING_SFX, txt), 2)
        if s > best:
            best, best_txt = s, txt
    if not any(e.kind == "dialogue" for e in first):
        best -= 0.1  # silent openings rarely stop a scroll on sound-on feeds
    if ep_start_bonus:
        best -= 0.15  # cold opens usually re-establish setting before the drama hits
    return _clip(best), best_txt


def _cliff(evs: list[Event], seg_end: float, all_ep: list[Event], is_ep_end: bool) -> tuple[float, str]:
    tail = [e for e in evs if e.start >= seg_end - TAIL_WINDOW]
    s, txt = 0.0, ""
    for e in tail:
        if e.kind == "sfx" and _n(STING_SFX, e.text):
            s += 0.25
        if _n(CUT_TO_BLACK, e.text):
            s += 0.15
        if e.kind == "action" and _n(PHYSICAL, e.text):
            s += 0.15
    lines = [e for e in evs if e.kind == "dialogue"]
    if lines:
        last = lines[-1]
        if last.start >= seg_end - TAIL_WINDOW * 1.5:
            txt = f'{last.speaker}: "{last.text}"'
            if last.text.endswith(("?", "...", "…", "—", "-")):
                s += 0.25
            if _n(THREAT_LINE, last.text):
                s += 0.2
            if _n(INTENSE_TAGS, " ".join(last.tags)):
                s += 0.15
            if _n(CALM_TAGS, " ".join(last.tags)):
                s -= 0.2
    # an action that is still happening when we cut = interrupted moment
    spanning = [e for e in all_ep if e.kind in ("action", "sfx") and e.start < seg_end - 0.3 < e.end - 0.3]
    if any(_n(PHYSICAL, e.text) or _n(STING_SFX, e.text) for e in spanning):
        s += 0.25
        if not txt:
            txt = next(e.text for e in spanning if _n(PHYSICAL, e.text) or _n(STING_SFX, e.text))
    if any(_n(RESOLUTION, e.text) for e in tail if e.kind != "dialogue"):
        s -= 0.25
    if is_ep_end:
        s += 0.15  # writers build each episode to end on a cliffhanger
    if not txt and tail:
        txt = tail[-1].text
    return _clip(s), txt


def _standalone(evs: list[Event], seg_start: float) -> tuple[float, list[str]]:
    flags = []
    lines = [e for e in evs if e.kind == "dialogue"]
    if len(lines) < 3:
        return 0.2, ["too little dialogue to follow"]
    s = 0.6
    speakers = {l.speaker for l in lines}
    if len(speakers) > 4:
        s -= 0.1 * (len(speakers) - 4)
        flags.append(f"{len(speakers)} speakers")
    opening = " ".join(l.text for l in lines[:3])
    if _n(BACKREF, opening):
        s -= 0.2
        flags.append("opens by referring to earlier events")
    if lines[0].text[:1].islower() or lines[0].text.startswith(("...", "…")):
        s -= 0.2
        flags.append("opens mid-sentence")
    body = " ".join(l.text for l in lines)
    s += 0.1 * min(_n(RELATION, body), 3)  # relationships make stakes legible fast
    scenes = {e.scene for e in evs}
    if len(scenes) > 1:
        s -= 0.15
        flags.append("crosses a scene change")
    return _clip(s), flags


def _intensity(evs: list[Event], dur: float) -> float:
    lines = [e for e in evs if e.kind == "dialogue"]
    hot = sum(1 for l in lines if _n(INTENSE_TAGS, " ".join(l.tags)))
    phys = sum(min(_n(PHYSICAL, e.text), 2) for e in evs if e.kind in ("action", "sfx", "camera"))
    calm = sum(1 for l in lines if _n(CALM_TAGS, " ".join(l.tags)))
    raw = (hot - 0.7 * calm) / max(len(lines), 1) * 0.6 + phys / max(dur, 1) * 10 * 0.08
    return _clip(raw)


def _duration_fit(d: float) -> float:
    if 28 <= d <= 45:
        return 1.0
    if d < 28:
        return _clip((d - MIN_LEN) / 8)
    return _clip(1 - (d - 45) / 15)


WEIGHTS = {"hook": 0.32, "cliffhanger": 0.30, "standalone": 0.18, "intensity": 0.14, "duration": 0.06}


def score_rules(c: Candidate, ep_by_num: dict[int, Episode]) -> Candidate:
    first_seg, first_evs = c.segments[0], c.events[0]
    last_seg, last_evs = c.segments[-1], c.events[-1]
    ep_last = ep_by_num[last_seg.episode]
    hook, hook_txt = _hook(first_evs, first_seg.start, first_seg.start < 1.0)
    cliff, cliff_txt = _cliff(last_evs, last_seg.end, ep_last.events,
                              abs(last_seg.end - ep_last.duration) < 1.0)
    if c.kind == "flash-forward":
        cliff = _clip(cliff + 0.3)  # we cut right before the beat the viewer was promised
        cliff_txt = cliff_txt or "cuts just before the teased moment happens"
    all_evs = [e for evs in c.events for e in evs]
    stand, flags = _standalone(all_evs, first_seg.start)
    inten = _intensity(all_evs, c.duration)
    c.scores = {"hook": round(hook, 3), "cliffhanger": round(cliff, 3), "standalone": round(stand, 3),
                "intensity": round(inten, 3), "duration": round(_duration_fit(c.duration), 3)}
    c.scores["rules"] = round(sum(WEIGHTS[k] * c.scores[k] for k in WEIGHTS), 4)
    c.hook, c.cliffhanger, c.flags = hook_txt, cliff_txt, flags
    c.why = _why(c)
    return c


def _why(c: Candidate) -> str:
    s = c.scores
    parts = []
    if s["hook"] >= 0.6:
        parts.append("opens on an immediate confrontation")
    if s["cliffhanger"] >= 0.6:
        parts.append("cuts before the outcome lands")
    if s["intensity"] >= 0.5:
        parts.append("high-conflict throughout")
    if s["standalone"] >= 0.85:
        parts.append("stakes are clear without backstory")
    if c.kind == "flash-forward":
        parts.insert(0, "teases the climax first, then rewinds to the setup")
    return ("; ".join(parts) or "moderate on every axis").capitalize() + "."


# --- flash-forward stitches ----------------------------------------------------------

def flash_forward_candidates(ep: Episode) -> list[Candidate]:
    """ReelShort-style ad: show the explosive beat first (2-5s), then rewind to the setup
    and cut right before the beat happens again. Two segments from one episode."""
    pts = cut_points(ep)
    hot = [e for e in ep.events
           if e.kind in ("action", "sfx") and _n(PHYSICAL, e.text) + _n(STING_SFX, e.text) >= 1]
    # Group back-to-back beats (strike -> gasp -> stagger) and start the teaser at the FIRST
    # one; otherwise the setup segment would already contain the payoff.
    clusters, cur = [], []
    for e in hot:
        if cur and e.start - max(x.end for x in cur) > 2.0:
            clusters.append(cur)
            cur = []
        cur.append(e)
    if cur:
        clusters.append(cur)
    beats = [c[0] for c in clusters if c[0].start > 15]
    out = []
    for b in beats:
        b_start = max([p for p in pts if p <= b.start + 0.05], default=None)
        b_end = min([p for p in pts if p >= b_start + 2.0 and p <= b_start + 6.0], default=None)
        if b_start is None or b_end is None:
            continue
        setups = [p for p in pts if 18 <= b_start - p <= 45]
        if not setups:
            continue
        s0 = min(setups, key=lambda p: abs((b_start - p) - 30))
        segs = [Segment(ep.number, b_start, b_end), Segment(ep.number, s0, b_start)]
        out.append(Candidate("flash-forward", segs, [events_in(ep, b_start, b_end),
                                                     events_in(ep, s0, b_start)]))
    return out
