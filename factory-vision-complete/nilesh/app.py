"""streamlit run app.py  (local only; never deploy publicly, footage is restricted)"""
from pathlib import Path
import pandas as pd, streamlit as st

DATA = Path("egohack-data/clips_480p")
clips = sorted(p.stem for p in DATA.glob("*.mp4"))
name = st.sidebar.selectbox("Clip", clips)
left, right = st.columns([3, 2])
left.video(str(DATA / f"{name}.mp4"))
csv = Path("outputs") / name / "actions.csv"
if csv.exists():
    right.dataframe(pd.read_csv(csv), use_container_width=True)
else:
    right.info(f"No timeline yet. Run: python run.py timeline {DATA / (name + '.mp4')}")
