"""Stage 2 (optional): an LLM re-judges only the shortlist from the rules stage.

The model is swappable with one flag. Every provider is a plain HTTPS call (no SDKs), so
plugging in whatever API the interview panel provides is a config change:

    --llm gemini     GEMINI_API_KEY      (free tier: Google AI Studio)
    --llm openai     OPENAI_API_KEY      + optional OPENAI_BASE_URL -> works for Groq,
                                          OpenRouter, Together, Ollama, vLLM, any OpenAI-compatible API
    --llm anthropic  ANTHROPIC_API_KEY
    --llm none       rules only (default; free, deterministic, offline)

The LLM gets ONLY the clip's own transcript - no logline, no show context - so it judges
the clip the way a cold viewer scrolling TikTok would.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODELS = {"gemini": "gemini-3.5-flash,gemini-flash-lite-latest", "openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5"}
CACHE_DIR = Path(".cache/llm")

PROMPT = """You are a performance-marketing editor for vertical microdrama apps (ReelShort/DramaBox style).
Below is the transcript of ONE candidate promo clip ({duration:.0f}s), cut from a show. Timestamps are
relative to the start of the clip. You have NOT seen the show - judge it as a stranger scrolling a feed.

{transcript}

Score the clip and answer with JSON only, no prose, exactly these keys:
{{
  "hook_score": 1-10,          // does something in the FIRST 3 SECONDS stop a scroll (slap, reveal, shocking line)?
  "hook": "the exact moment/line that hooks, quoted from the transcript",
  "cliffhanger_score": 1-10,   // does it END unresolved, so the viewer must know what happens next?
  "cliffhanger": "the exact final unresolved moment, quoted",
  "standalone_score": 1-10,    // could a stranger follow who is who and what is at stake?
  "cold_viewer_summary": "one sentence: what a stranger thinks is happening",
  "confusing_references": ["names/events a stranger would not understand"],
  "why": "one line on why this works as an ad (or why not)",
  "trim_suggestion": "optional: e.g. 'start 4s later at the slap' or ''"
}}
Be harsh: most clips are a 4-6. Only quote text that appears in the transcript."""


class LLMError(RuntimeError):
    pass


def _post(url: str, headers: dict, body: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 4:  # free tiers rate-limit hard
                time.sleep(2 ** attempt * 3)
                continue
            raise LLMError(f"{e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:  # thinking models can be slow
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(str(e)) from e
    raise LLMError("retries exhausted")


def _complete(provider: str, model: str, prompt: str) -> str:
    """`model` may be a comma-separated fallback chain, e.g. 'big-model,small-model'."""
    chain = [m.strip() for m in model.split(",") if m.strip()]
    for i, m in enumerate(chain):
        try:
            return _complete_one(provider, m, prompt)
        except LLMError:
            if i == len(chain) - 1:
                raise
    raise LLMError("empty model chain")


def _complete_one(provider: str, model: str, prompt: str) -> str:
    if provider == "gemini":
        key = os.environ["GEMINI_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
        r = _post(url, {"Content-Type": "application/json"}, body)
        return r["candidates"][0]["content"]["parts"][0]["text"]
    if provider == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'ollama')}"}
        body = {"model": model, "temperature": 0.2, "messages": [{"role": "user", "content": prompt}]}
        r = _post(f"{base}/chat/completions", headers, body)
        return r["choices"][0]["message"]["content"]
    if provider == "anthropic":
        headers = {"Content-Type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                   "anthropic-version": "2023-06-01"}
        body = {"model": model, "max_tokens": 800, "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}]}
        r = _post("https://api.anthropic.com/v1/messages", headers, body)
        return r["content"][0]["text"]
    raise LLMError(f"unknown provider {provider}")


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError(f"no JSON in response: {text[:200]}")
    return json.loads(m.group(0))


def judge(transcript: str, duration: float, provider: str, model: str | None = None) -> dict:
    model = model or DEFAULT_MODELS[provider]
    prompt = PROMPT.format(transcript=transcript, duration=duration)
    key = hashlib.sha256(f"{provider}|{model}|{prompt}".encode()).hexdigest()[:24]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out = _parse_json(_complete(provider, model, prompt))
    for k in ("hook_score", "cliffhanger_score", "standalone_score"):
        out[k] = max(1, min(10, float(out.get(k, 5))))
    out["model"] = f"{provider}/{model}"
    out["hallucinated_quotes"] = _check_quotes(out, transcript)
    cache.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _check_quotes(out: dict, transcript: str) -> list[str]:
    """Guardrail: the hook/cliffhanger the model quotes must actually be in the clip."""
    bad = []
    norm = re.sub(r"\W+", " ", transcript.lower())
    for k in ("hook", "cliffhanger"):
        q = re.sub(r"\W+", " ", str(out.get(k, "")).lower()).strip()
        words = q.split()
        # accept if any 4-word run from the quote is present (models paraphrase speaker labels)
        runs = [" ".join(words[i:i + 4]) for i in range(max(1, len(words) - 3))]
        if words and not any(r in norm for r in runs):
            bad.append(k)
    return bad
