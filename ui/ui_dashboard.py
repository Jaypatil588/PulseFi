#!/usr/bin/env python3
"""Streamlit dashboard for live PulseFi predictions."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="PulseFi Live", page_icon="📶", layout="wide")

st.markdown(
    """
<style>
.main {background: linear-gradient(180deg,#f7fbff 0%,#eef6ff 100%);}
.card {
  border: 1px solid #dbe8ff;
  border-radius: 12px;
  padding: 12px 14px;
  background: #ffffff;
}
.title {
  font-size: 0.86rem;
  color: #36547a;
  font-weight: 600;
}
.value {
  font-size: 1.35rem;
  color: #132b4d;
  font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("PulseFi Live Monitor")
st.caption("Stable-vs-Human classification + BPM prediction (human only)")

default_csv = "/Users/senpai/Desktop/projects/pulseFi/runtime/live_predictions.csv"
csv_path = st.text_input("Predictions CSV", value=default_csv)
refresh_sec = st.slider("Refresh (seconds)", min_value=1, max_value=10, value=2)

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.utcnow()

if st.button("Refresh now"):
    st.session_state.last_refresh = datetime.utcnow()

if not os.path.exists(csv_path):
    st.warning(f"File not found: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)
if df.empty:
    st.warning("No predictions yet. Start run_live_inference.py first.")
    st.stop()

df["time"] = pd.to_datetime(df["rx_ts_us"], unit="us")
df = df.sort_values("rx_ts_us")
latest = df.iloc[-1]

cls_text = "HUMAN" if int(latest["class_pred"]) == 1 else "STABLE"
cls_color = "#0a7f3f" if cls_text == "HUMAN" else "#7a2e0d"

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(
    f"<div class='card'><div class='title'>Class</div><div class='value' style='color:{cls_color}'>{cls_text}</div></div>",
    unsafe_allow_html=True,
)
c2.markdown(
    f"<div class='card'><div class='title'>Human Probability</div><div class='value'>{float(latest['class_prob_human']):.3f}</div></div>",
    unsafe_allow_html=True,
)
pred_bpm = latest["pred_bpm"] if pd.notna(latest["pred_bpm"]) and str(latest["pred_bpm"]) != "" else "-"
c3.markdown(
    f"<div class='card'><div class='title'>Pred BPM</div><div class='value'>{pred_bpm}</div></div>",
    unsafe_allow_html=True,
)
sensor_bpm = latest["sensor_bpm"] if pd.notna(latest["sensor_bpm"]) and str(latest["sensor_bpm"]) != "" else "-"
c4.markdown(
    f"<div class='card'><div class='title'>Sensor BPM</div><div class='value'>{sensor_bpm}</div></div>",
    unsafe_allow_html=True,
)
c5.markdown(
    f"<div class='card'><div class='title'>Rows</div><div class='value'>{len(df)}</div></div>",
    unsafe_allow_html=True,
)

left, right = st.columns([1.25, 1])

with left:
    fig_prob = go.Figure()
    fig_prob.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["class_prob_human"],
            mode="lines",
            name="P(human)",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig_prob.add_hline(y=0.5, line_dash="dash", line_color="#999")
    fig_prob.update_layout(
        title="Human Probability",
        xaxis_title="Time",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=20, t=45, b=20),
        height=320,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    fig_bpm = go.Figure()
    if "pred_bpm" in df:
        fig_bpm.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["pred_bpm"],
                mode="lines",
                name="Pred BPM",
                line=dict(color="#0f8a5f", width=2),
            )
        )
    if "sensor_bpm" in df:
        fig_bpm.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["sensor_bpm"],
                mode="lines",
                name="Sensor BPM",
                line=dict(color="#d62728", width=2),
            )
        )
    fig_bpm.update_layout(
        title="BPM Prediction vs Sensor",
        xaxis_title="Time",
        yaxis_title="BPM",
        margin=dict(l=20, r=20, t=45, b=20),
        height=320,
    )
    st.plotly_chart(fig_bpm, use_container_width=True)

with right:
    class_counts = df["class_pred"].value_counts().sort_index()
    stable_count = int(class_counts.get(0, 0))
    human_count = int(class_counts.get(1, 0))

    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=["Stable", "Human"],
                values=[stable_count, human_count],
                hole=0.55,
                marker=dict(colors=["#b96d35", "#2a9d8f"]),
            )
        ]
    )
    fig_pie.update_layout(
        title="Predicted Class Distribution",
        margin=dict(l=20, r=20, t=45, b=20),
        height=320,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Latest Rows")
    show_cols = ["time", "class_prob_human", "class_pred", "pred_bpm", "sensor_bpm"]
    st.dataframe(df[show_cols].tail(15), use_container_width=True, height=315)

st.caption(f"Last refresh: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
st.caption("Tip: keep this tab open while run_live_inference.py writes the CSV.")

# Cheap auto-refresh without extra dependencies.
st.markdown(
    f"""
    <script>
      setTimeout(function() {{
        window.location.reload();
      }}, {int(refresh_sec * 1000)});
    </script>
    """,
    unsafe_allow_html=True,
)
