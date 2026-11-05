"""Minimal web UI: upload a .zip of screenplays, get ranked promos.  Run: streamlit run app.py"""
import io
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder.pipeline import find_promos, quality_checks, write_outputs

st.set_page_config(page_title="Promo Finder", layout="wide")
st.title("Promo Finder")
st.caption("Upload a show's screenplays (.zip of episode .txt files). Get ranked 20-60s promo clips.")

with st.sidebar:
    llm = st.selectbox("Judge", ["none", "gemini", "openai", "anthropic"],
                       help="'none' = rules only (free, instant). Others re-judge the shortlist.")
    model = st.text_input("Model (optional)", "")
    key = st.text_input("API key", type="password") if llm != "none" else ""
    base = st.text_input("OpenAI-compatible base URL", "") if llm == "openai" else ""
    top = st.slider("Promos to return", 5, 50, 25)

up = st.file_uploader("Screenplays .zip", type="zip")
if up and st.button("Find promos", type="primary"):
    env = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    if key:
        os.environ[env[llm]] = key
    if base:
        os.environ["OPENAI_BASE_URL"] = base
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(io.BytesIO(up.read())).extractall(tmp)
        status = st.status("Working...")
        res = find_promos(tmp, top=top, llm=llm, model=model or None, log=status.write)
        status.update(label="Done", state="complete")
        out = write_outputs(res, Path(tmp) / "out")
        report = (out / "report.md").read_text(encoding="utf-8")
        csv_bytes = (out / "promos.csv").read_bytes()
    by_num = {e.number: e for e in res["episodes"]}
    rows = []
    for i, c in enumerate(res["post_order"], 1):
        rows.append({"#": i, "segments": " | ".join(f"Ep{s.episode} {s.start:.1f}-{s.end:.1f}s" for s in c.segments),
                     "len": round(c.duration, 1), "hook": c.hook, "cliffhanger": c.cliffhanger, "why": c.why,
                     "score": c.scores["post_priority"], "issues": "; ".join(quality_checks(c, by_num) + c.flags)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.download_button("Download promos.csv", csv_bytes, "promos.csv")
    st.download_button("Download report.md", report, "report.md")
    for i, c in enumerate(res["post_order"], 1):
        with st.expander(f"#{i}  {rows[i-1]['segments']}  ({c.duration:.0f}s)"):
            st.code(c.transcript(), language=None)
