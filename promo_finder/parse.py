"""Parse Kuku-style microdrama screenplays into a flat list of timed events.

Expected format (one file per episode):

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    SCENE 01
    Scene heading: INT. CATHEDRAL - DAY
    **CAMERA:**
    Start: (00:00.00) | End: (00:04.80)
    Symmetrical framing on the heavy church doors...
    **LADY JULIANA (Confused, Breathless):**
    Start: (00:13.50) | End: (00:14.30)
    "Why?"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADER_RE = re.compile(r"^\*\*(?P<label>.+?):\*\*\s*$")
TIME_RE = re.compile(r"Start:\s*\((?P<s>[\d:.]+)\)\s*\|\s*End:\s*\((?P<e>[\d:.]+)\)")
EP_RE = re.compile(r"EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)
LOGLINE_RE = re.compile(r"EPISODE LOGLINE:\s*(.+)")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)")
HEADING_RE = re.compile(r"^Scene heading:\s*(.+)")


def to_seconds(ts: str) -> float:
    """'01:53.30' -> 113.3. Also accepts 'hh:mm:ss.xx'."""
    parts = [float(p) for p in ts.split(":")]
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return round(total, 2)


def fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Event:
    episode: int
    scene: int
    kind: str          # camera | action | sfx | dialogue
    start: float
    end: float
    text: str
    speaker: str = ""
    tags: list[str] = field(default_factory=list)  # emotion tags from "(Cold, Commanding)"

    def line(self, offset: float = 0.0) -> str:
        t = f"[{fmt(self.start - offset)}]"
        if self.kind == "dialogue":
            tag = f" ({', '.join(self.tags)})" if self.tags else ""
            return f"{t} {self.speaker}{tag}: {self.text}"
        return f"{t} {self.kind.upper()}: {self.text}"


@dataclass
class Episode:
    number: int
    logline: str
    path: str
    events: list[Event]
    scene_headings: dict[int, str]

    @property
    def duration(self) -> float:
        return max((e.end for e in self.events), default=0.0)


def _classify(label: str) -> tuple[str, str, list[str]]:
    up = label.upper().strip()
    if up == "CAMERA":
        return "camera", "", []
    if up == "ACTION":
        return "action", "", []
    if up.startswith("SOUND"):
        return "sfx", "", []
    # Dialogue: "LADY JULIANA (Confused, Breathless)" or "MATRIARCH (O.S.) (Firm)"
    groups = re.findall(r"\(([^)]*)\)", label)
    speaker = re.sub(r"\s*\([^)]*\)", "", label).strip()
    tags = []
    for g in groups:
        for t in g.split(","):
            t = t.strip()
            if t and t.upper() not in {"O.S.", "V.O.", "CONT'D", "O.C."}:
                tags.append(t)
    return "dialogue", speaker, tags


def parse_episode(path: str | Path) -> Episode:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    m = EP_RE.search(text)
    number = int(m.group(1)) if m else -1
    lm = LOGLINE_RE.search(text)
    logline = lm.group(1).strip() if lm else ""

    events: list[Event] = []
    headings: dict[int, str] = {}
    scene = 1
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if sm := SCENE_RE.match(line):
            scene = int(sm.group(1))
        elif hm := HEADING_RE.match(line):
            headings[scene] = hm.group(1).strip()
        elif h := HEADER_RE.match(line):
            kind, speaker, tags = _classify(h.group("label"))
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            tm = TIME_RE.search(lines[j]) if j < len(lines) else None
            if tm:
                body = []
                k = j + 1
                while k < len(lines) and lines[k].strip() and not HEADER_RE.match(lines[k].strip()):
                    body.append(lines[k].strip())
                    k += 1
                body_text = " ".join(body).strip().strip('"').strip("“”")
                events.append(Event(number, scene, kind, to_seconds(tm.group("s")),
                                    to_seconds(tm.group("e")), body_text, speaker, tags))
                i = k
                continue
        i += 1
    events.sort(key=lambda e: (e.start, e.end))
    return Episode(number, logline, str(path), events, headings)


def load_show(folder: str | Path) -> list[Episode]:
    """Load every *.txt screenplay under a folder (recursively), sorted by episode number."""
    eps = [parse_episode(p) for p in sorted(Path(folder).rglob("*.txt"))]
    eps = [e for e in eps if e.events]
    eps.sort(key=lambda e: e.number)
    return eps
