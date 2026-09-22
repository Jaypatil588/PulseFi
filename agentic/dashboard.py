"""Streamlit dashboard for committed agent state and user messages."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agentic.db import MemoryDB
from agentic.models import DISCLAIMER
from agentic.worker import DEFAULT_DB


st.set_page_config(
    page_title="PulseFi Agent Monitor",
    page_icon="",
    layout="wide",
)

st.title("PulseFi Agent Monitor")
st.caption(DISCLAIMER)

db_path = Path(
    st.sidebar.text_input("Agent database", value=str(DEFAULT_DB))
)
st.sidebar.caption("Refreshes from committed SQLite state every second.")


def _snapshot() -> tuple[dict, list[dict]]:
    db = MemoryDB(db_path)
    try:
        return db.dashboard_snapshot(), db.dashboard_history()
    finally:
        db.close()


def _acknowledge(action_id: int) -> None:
    db = MemoryDB(db_path)
    try:
        db.acknowledge_action(action_id, int(time.time() * 1_000_000))
    finally:
        db.close()


@st.fragment(run_every=1.0)
def live_panel() -> None:
    if not db_path.exists():
        st.warning(f"Waiting for agent database: {db_path}")
        return
    snapshot, history = _snapshot()
    latest = snapshot["latest"]
    if latest is None:
        st.info("Waiting for the first committed agent decision.")
        return

    age_s = max(
        0.0, (int(time.time() * 1_000_000) - int(latest["ts_us"])) / 1e6
    )
    if age_s > 5.0:
        st.error(
            f"Input is stale: last observation was {age_s:.1f} seconds ago."
        )
    if not bool(latest["reasoning_available"]):
        status = str(latest["reasoning_status"])
        message = (
            "Groq reasoning was not scheduled for this tick."
            if status == "not_scheduled"
            else (
                "Groq reasoning is disabled because no API credential is "
                "configured."
                if status == "disabled"
                else "The Groq request failed for this tick."
            )
        )
        st.info(
            f"{message} The displayed final state is the deterministic "
            "measurement guard."
        )

    final_state = str(latest["final_state"])
    columns = st.columns(5)
    columns[0].metric("Final state", final_state.upper())
    columns[1].metric(
        "BPM",
        "—" if latest["bpm"] is None else f"{float(latest['bpm']):.1f}",
    )
    columns[2].metric("Guard", str(latest["guard_state"]).upper())
    columns[3].metric(
        "LLM",
        (
            str(latest["llm_state"]).upper()
            if latest["llm_state"] is not None
            else "NOT RUN"
        ),
    )
    columns[4].metric(
        "Confidence", f"{float(latest['confidence']):.0%}"
    )

    unacknowledged = [
        action
        for action in snapshot["actions"]
        if action["acknowledged_us"] is None
    ]
    if unacknowledged:
        action = unacknowledged[0]
        if action["severity"] == "urgent":
            st.error(action["message"])
        elif action["severity"] in {"watch", "recovering"}:
            st.warning(action["message"])
        else:
            st.success(action["message"])
        if st.button(
            "Acknowledge message",
            key=f"ack_{action['id']}",
            type="primary",
        ):
            _acknowledge(int(action["id"]))
            st.rerun(scope="fragment")

    left, right = st.columns([1.4, 1.0])
    with left:
        st.subheader("Agent interpretation")
        st.write(f"**Trend:** {latest['trend']}")
        st.write(f"**Forecast:** {latest['forecast']}")
        st.write(f"**Feedback:** {latest['feedback']}")
        st.write(f"**Suggestion:** {latest['suggestion']}")
        st.write("**Evidence:**")
        for evidence in latest["evidence_json"]:
            st.write(f"- {evidence}")
    with right:
        st.subheader("Current context")
        st.write(f"**Source:** `{latest['source']}`")
        st.write(f"**Activity:** `{latest['activity']}`")
        st.write(f"**Presence:** {float(latest['presence']):.3f}")
        st.write(f"**Model:** `{latest['model']}`")
        st.write(
            f"**Tokens:** {int(latest['total_tokens'])} "
            f"({int(latest['prompt_tokens'])} in / "
            f"{int(latest['completion_tokens'])} out)"
        )

    if history:
        frame = pd.DataFrame(history)
        frame["time"] = pd.to_datetime(frame["ts_us"], unit="us")
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame["bpm"],
                mode="lines+markers",
                name="LSTM BPM",
            )
        )
        baseline_raw = snapshot["semantic_memory"].get("resting_bpm")
        if baseline_raw is not None:
            figure.add_hline(
                y=float(baseline_raw),
                line_dash="dash",
                annotation_text="Personal baseline",
            )
        figure.update_layout(
            title="Committed heart-rate history",
            xaxis_title="Time",
            yaxis_title="BPM",
            height=360,
        )
        st.plotly_chart(figure, width="stretch")

    episode_tab, memory_tab, action_tab, integration_tab, input_tab = st.tabs(
        [
            "Episodes",
            "Memory",
            "Messages",
            "Integration outbox",
            "Input audit",
        ]
    )
    with episode_tab:
        if snapshot["episodes"]:
            st.dataframe(
                pd.DataFrame(snapshot["episodes"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No episodes recorded.")
    with memory_tab:
        st.json(snapshot["semantic_memory"])
    with action_tab:
        if snapshot["actions"]:
            st.dataframe(
                pd.DataFrame(snapshot["actions"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No messages recorded.")
    with integration_tab:
        st.caption(
            "Outbox events feed the optional local JSONL transport and future "
            "MCP/export adapters. External delivery is not claimed."
        )
        if snapshot["integration_events"]:
            st.write("**Outbox events**")
            st.dataframe(
                pd.DataFrame(snapshot["integration_events"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No integration events recorded.")
        if snapshot["delivery_attempts"]:
            st.write("**Delivery attempts**")
            st.dataframe(
                pd.DataFrame(snapshot["delivery_attempts"]),
                width="stretch",
                hide_index=True,
            )
    with input_tab:
        if snapshot["stream_events"]:
            st.dataframe(
                pd.DataFrame(snapshot["stream_events"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No input rows have been rejected.")


live_panel()
