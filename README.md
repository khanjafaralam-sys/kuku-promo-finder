# Promo Finder

Reads a microdrama screenplay pack and returns the best 20 to 60-second promo clips: episode,
start/end of every segment, the hook, the cliffhanger, why it works, and the order to post them in.

## Run it

```bash
python find_promos.py path/to/show.zip --out outputs/run1
```

That's all. The CLI uses only the Python 3.10+ standard library. It takes about 12 seconds for 50 episodes and costs nothing.

With an LLM judge (optional; it re-scores only the top 40 clips):

```bash
# Gemini free tier (key from aistudio.google.com)
set GEMINI_API_KEY=...        # macOS/Linux: export GEMINI_API_KEY=...
python find_promos.py show.zip --llm gemini --out outputs/gemini
# default model chain: gemini-3.5-flash, falling back to gemini-flash-lite-latest if overloaded.
# Override with --model, e.g. --model gemini-3.8-flash,gemini-3.5-flash

# Any OpenAI-compatible API: OpenAI, Groq, OpenRouter, a local Ollama, or an API you provide
set OPENAI_BASE_URL=https://api.groq.com/openai/v1
set OPENAI_API_KEY=...
python find_promos.py show.zip --llm openai --model llama-3.3-70b-versatile

python find_promos.py show.zip --llm anthropic --model <model-id>
```

Web UI: `pip install -r requirements.txt` then `streamlit run app.py`.

## Outputs (in `--out`)

| File | What it is |
|---|---|
| `report.md` | Post-first ranking table, auto-flagged quality issues, and the full transcript of every clip |
| `promos.csv` / `promos.json` | Machine-readable ranking with every sub-score |
| `best_per_episode.csv` | The best 1 to 2 clips for **every** episode, for promoting a specific one |
| `editor_review_sheet.csv` | Keep/trim/reject sheet for the human editor |

## How it works

```
screenplay .txt ──parse──▶ timed events (camera / action / sfx / dialogue + emotion tags)
                 ──cut───▶ every 20-60s window between CAMERA shot boundaries that doesn't split a line
                           + "flash-forward" stitches (climax beat first, then the setup, cut before it lands)
                 ──rules─▶ hook (first 3s) · cliffhanger (last 6s) · standalone · intensity · length
                 ──select▶ best 2 non-overlapping per episode
                 ──LLM?──▶ cold-viewer re-judge of the top 40 (sees ONLY the clip, never the logline)
                 ──rank──▶ quality score × funnel weight (earlier episodes first, fewer spoilers)
```

* `promo_finder/parse.py`: screenplay parser
* `promo_finder/score.py`: candidate generation and rule scoring (the lexicons at the top are tunable per genre)
* `promo_finder/llm.py`: provider-agnostic LLM judge, response cache, and a hallucinated-quote guardrail
* `promo_finder/pipeline.py`: selection, ranking, quality checks, and output files

## Confidentiality

The input pack belongs to Kuku. `.gitignore` excludes `data/`, `*.zip` and `outputs/`, so don't
force-add them to a public repo. Share raw outputs through a private Drive link instead.
