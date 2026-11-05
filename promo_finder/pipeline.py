"""End-to-end: screenplays -> scored candidates -> selected, ranked promos -> files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import llm as llm_mod
from .parse import Episode, fmt, load_show
from .score import Candidate, cut_points, flash_forward_candidates, score_rules, single_candidates

PER_EPISODE = 2       # max promos kept per episode (diversity)
LLM_WEIGHT = 0.6      # final = (1-w)*rules + w*llm when an LLM is used


def _overlaps(a: Candidate, b: Candidate) -> bool:
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode == sb.episode and sa.start < sb.end - 1 and sb.start < sa.end - 1:
                return True
    return False


def select(cands: list[Candidate], k: int) -> list[Candidate]:
    """Greedy non-overlapping selection (like NMS in object detection)."""
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.total, reverse=True):
        if not any(_overlaps(c, o) for o in kept):
            kept.append(c)
        if len(kept) >= k:
            break
    return kept


def find_promos(folder: str, top: int = 25, llm: str = "none", model: str | None = None,
                shortlist: int = 40, log=print) -> dict:
    eps = load_show(folder)
    if not eps:
        raise SystemExit(f"No screenplays found in {folder}")
    by_num = {e.number: e for e in eps}
    log(f"Loaded {len(eps)} episodes, {sum(len(e.events) for e in eps)} events.")

    per_ep: list[Candidate] = []
    n_scored = 0
    for ep in eps:
        cands = single_candidates(ep) + flash_forward_candidates(ep)
        for c in cands:
            score_rules(c, by_num)
        n_scored += len(cands)
        per_ep += select(cands, PER_EPISODE)
    log(f"Scored {n_scored} candidate clips; kept {len(per_ep)} (best {PER_EPISODE} per episode).")

    if llm != "none":
        pool = sorted(per_ep, key=lambda c: c.total, reverse=True)[:shortlist]
        log(f"LLM ({llm}) judging top {len(pool)} ...")
        for i, c in enumerate(pool, 1):
            try:
                c.llm = llm_mod.judge(c.transcript(), c.duration, llm, model)
            except Exception as e:  # keep going; a failed call just leaves the rules score
                c.flags.append(f"llm error: {e}"[:160])
                continue
            j = c.llm
            llm_score = (0.4 * j["hook_score"] + 0.35 * j["cliffhanger_score"] + 0.25 * j["standalone_score"]) / 10
            if j.get("hallucinated_quotes"):
                llm_score *= 0.8
                c.flags.append("LLM quoted text not in clip: " + ",".join(j["hallucinated_quotes"]))
            c.scores["llm"] = round(llm_score, 4)
            c.scores["final"] = round((1 - LLM_WEIGHT) * c.scores["rules"] + LLM_WEIGHT * llm_score, 4)
            if abs(c.scores["rules"] - llm_score) > 0.25:
                c.flags.append("rules and LLM disagree - human check")
            bad = j.get("hallucinated_quotes", [])
            if j.get("hook") and "hook" not in bad:
                c.hook = j["hook"]
            if j.get("cliffhanger") and "cliffhanger" not in bad:
                c.cliffhanger = j["cliffhanger"]
            if j.get("why"):
                c.why = j["why"]
            if i % 10 == 0:
                log(f"  {i}/{len(pool)}")
        judged = [c for c in pool if "llm" in c.scores]
        ranked = sorted(judged, key=lambda c: c.total, reverse=True)[:top]
    else:
        ranked = sorted(per_ep, key=lambda c: c.total, reverse=True)[:top]

    n_eps = max(by_num)
    for c in ranked:
        ep = c.segments[0].episode
        # Post-first order: a new viewer lands on episode 1, so early-episode promos convert
        # faster and don't spoil the show's central twist.
        funnel = 1.0 if ep <= 10 else 1.0 - 0.3 * (ep - 10) / max(n_eps - 10, 1)
        c.scores["post_priority"] = round(c.total * funnel, 4)
        if ep > n_eps * 0.5:
            c.flags.append("spoiler risk: from second half of the show")
    post_order = sorted(ranked, key=lambda c: c.scores["post_priority"], reverse=True)

    return {"episodes": eps, "ranked": ranked, "post_order": post_order, "per_episode": per_ep, "llm": llm,
            "model": model or llm_mod.DEFAULT_MODELS.get(llm, "-"), "n_scored": n_scored}


# --- quality checks ------------------------------------------------------------------

def quality_checks(c: Candidate, by_num: dict[int, Episode]) -> list[str]:
    issues = []
    if not 20 <= c.duration <= 60.5:
        issues.append(f"duration {c.duration:.1f}s out of 20-60s")
    for seg in c.segments:
        ep = by_num[seg.episode]
        for t in (seg.start, seg.end):
            if any(d.kind == "dialogue" and d.start < t - 0.15 and d.end > t + 0.15 for d in ep.events):
                issues.append(f"cuts mid-line at {fmt(t)}")
    first = [e for e in c.events[0] if e.start < c.segments[0].start + 3 and e.kind != "camera"]
    if not first:
        issues.append("nothing but camera in first 3s")
    if c.scores["hook"] < 0.35:
        issues.append("weak hook")
    if c.scores["cliffhanger"] < 0.35:
        issues.append("weak cliffhanger")
    return issues


# --- writers -------------------------------------------------------------------------

def _row(rank: int, c: Candidate) -> dict:
    return {
        "rank": rank, "type": c.kind,
        "episodes": "+".join(sorted({str(s.episode) for s in c.segments})),
        "segments": " | ".join(f"Ep{s.episode} {fmt(s.start)}-{fmt(s.end)}" for s in c.segments),
        "duration_s": round(c.duration, 1),
        "hook": c.hook, "cliffhanger": c.cliffhanger, "why": c.why,
        **{f"score_{k}": v for k, v in c.scores.items()},
        "flags": "; ".join(c.flags),
    }


def write_outputs(res: dict, out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_num = {e.number: e for e in res["episodes"]}
    ranked, post = res["ranked"], res["post_order"]

    rows = [_row(i, c) for i, c in enumerate(ranked, 1)]
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

    # every episode's best clip(s), so an editor promoting a specific episode has a starting point
    ep_rows = [_row(0, c) for c in sorted(res["per_episode"], key=lambda c: (c.segments[0].episode, -c.total))]
    with open(out / "best_per_episode.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in ep_rows for k in r)))
        w.writeheader()
        w.writerows(ep_rows)

    js = []
    for i, c in enumerate(ranked, 1):
        js.append({"rank": i, "type": c.kind, "segments": [s.as_dict() for s in c.segments],
                   "duration_s": round(c.duration, 1), "hook": c.hook, "cliffhanger": c.cliffhanger,
                   "why": c.why, "scores": c.scores, "flags": c.flags, "llm": c.llm,
                   "quality_issues": quality_checks(c, by_num), "transcript": c.transcript()})
    (out / "promos.json").write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")

    # human-in-the-loop sheet: editor marks keep/reject in 10 minutes instead of watching 50 episodes
    with open(out / "editor_review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["post_order", "segments", "duration_s", "hook", "cliffhanger",
                    "editor_verdict (keep/trim/reject)", "editor_notes"])
        for i, c in enumerate(post, 1):
            w.writerow([i, _row(0, c)["segments"], round(c.duration, 1), c.hook, c.cliffhanger, "", ""])

    md = [f"# Promo report\n",
          f"Episodes: {len(res['episodes'])} | candidates scored: {res['n_scored']} | "
          f"judge: rules{' + ' + res['llm'] + '/' + res['model'] if res['llm'] != 'none' else ' only'}\n",
          "## Post-first order\n",
          "| # | Segments | Len | Hook | Cliffhanger | Why | Score |", "|---|---|---|---|---|---|---|"]
    for i, c in enumerate(post, 1):
        r = _row(i, c)
        esc = lambda s: str(s).replace("|", "/").replace("\n", " ")[:140]
        md.append(f"| {i} | {r['segments']} | {r['duration_s']}s | {esc(c.hook)} | {esc(c.cliffhanger)} | "
                  f"{esc(c.why)} | {c.scores['post_priority']:.2f} |")
    md.append("\n## Quality issues flagged automatically\n")
    for i, c in enumerate(post, 1):
        iss = quality_checks(c, by_num) + c.flags
        if iss:
            md.append(f"- #{i} ({_row(i, c)['segments']}): " + "; ".join(iss))
    covered = {s.episode for c in ranked for s in c.segments}
    md.append(f"\nEpisodes with no promo in the top {len(ranked)}: "
              f"{', '.join(str(n) for n in sorted(by_num) if n not in covered)}")
    md.append("\n## Full clip transcripts\n")
    for i, c in enumerate(post, 1):
        md.append(f"### #{i}: {_row(i, c)['segments']} ({c.duration:.1f}s, {c.kind})\n")
        md.append("```\n" + c.transcript() + "\n```\n")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    return out
