"""Find the best promo clips in a folder (or .zip) of microdrama screenplays.

    python find_promos.py path/to/show_folder_or_zip --out outputs/run1
    python find_promos.py show.zip --llm gemini --out outputs/gemini     # needs GEMINI_API_KEY
    python find_promos.py show.zip --llm openai --model llama-3.3-70b-versatile   # e.g. Groq via OPENAI_BASE_URL
"""
import argparse
import tempfile
import zipfile
from pathlib import Path

from promo_finder.pipeline import find_promos, write_outputs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder of episode .txt files, or a .zip of them")
    ap.add_argument("--out", default="outputs/latest")
    ap.add_argument("--top", type=int, default=25, help="number of promos to output")
    ap.add_argument("--llm", default="none", choices=["none", "gemini", "openai", "anthropic"])
    ap.add_argument("--model", default=None, help="override the provider's default model")
    ap.add_argument("--shortlist", type=int, default=40, help="how many rule-ranked clips the LLM re-judges")
    a = ap.parse_args()

    src = Path(a.input)
    with tempfile.TemporaryDirectory() as tmp:
        if src.suffix.lower() == ".zip":
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            src = Path(tmp)
        res = find_promos(str(src), top=a.top, llm=a.llm, model=a.model, shortlist=a.shortlist)
    out = write_outputs(res, a.out)
    print(f"\nTop 5 to post first:")
    for i, c in enumerate(res["post_order"][:5], 1):
        segs = " + ".join(f"Ep{s.episode} {s.start:.1f}-{s.end:.1f}s" for s in c.segments)
        print(f" {i}. {segs} ({c.duration:.0f}s)  hook: {c.hook[:70]}")
    print(f"\nWrote {out}/report.md, promos.csv, promos.json, editor_review_sheet.csv")


if __name__ == "__main__":
    main()
