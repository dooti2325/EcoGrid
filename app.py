"""
EcoGrid-OpenEnv Streamlit Dashboard

A professional control-room dashboard for visualizing the RL environment and
comparing random, heuristic, and trained agents.
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from baseline import heuristic_agent, load_trained_model, local_llm_agent
from env.environment import EcoGridEnv
from models.schemas import GridAction


st.set_page_config(page_title="EcoGrid OpenEnv", layout="wide")

PRIMARY = "#27c3bd"
ACCENT = "#6ea8fe"
SUCCESS = "#45d483"
WARNING = "#f6c85f"
DANGER = "#f05252"
PAPER = "rgba(0,0,0,0)"
GRID = "rgba(148, 163, 184, 0.18)"
TEXT = "#e6edf7"
MUTED = "#9aa8bd"


st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --panel: #141d2b;
        --panel-soft: #192437;
        --line: rgba(148, 163, 184, 0.18);
        --text: #e6edf7;
        --muted: #9aa8bd;
        --primary: #27c3bd;
        --accent: #6ea8fe;
        --danger: #f05252;
    }

    .stApp {
        background:
            radial-gradient(circle at 24% 0%, rgba(39, 195, 189, 0.10), transparent 28rem),
            linear-gradient(135deg, #09111f 0%, #101827 48%, #0b1220 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1540px;
        padding: 2rem 2.2rem 2.6rem;
    }

    [data-testid="stSidebar"] {
        background: #111a28;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] .block-container,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 1rem;
    }

    [data-testid="stSidebar"] h1 {
        color: var(--text);
        font-size: 1.25rem;
        letter-spacing: 0;
        margin-bottom: 0.25rem;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: var(--text);
    }

    .hero {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(20, 29, 43, 0.96), rgba(17, 26, 40, 0.86));
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    }

    .eyebrow {
        color: var(--primary);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .hero h1 {
        color: var(--text);
        font-size: clamp(1.9rem, 3vw, 3.1rem);
        font-weight: 800;
        letter-spacing: 0;
        line-height: 1.05;
        margin: 0;
    }

    .hero p {
        color: var(--muted);
        font-size: 1rem;
        margin: 0.6rem 0 0;
        max-width: 760px;
    }

    .kpi-card {
        min-height: 104px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(20, 29, 43, 0.88);
        padding: 1rem;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .kpi-value {
        color: var(--text);
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 0.35rem;
    }

    .kpi-note {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 8px;
        background: rgba(20, 29, 43, 0.88);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 1rem 1rem 0.85rem;
    }

    .panel-title {
        color: var(--text);
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.75rem;
        margin-bottom: 0.85rem;
    }

    .panel-title strong {
        font-size: 1.02rem;
    }

    .panel-title span {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(25, 36, 55, 0.76);
        padding: 0.85rem 1rem;
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary);
        font-size: 1.75rem !important;
        font-weight: 800 !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 2.7rem;
        border: 1px solid rgba(39, 195, 189, 0.4);
        border-radius: 8px;
        background: #1faea9;
        color: #06111f;
        font-weight: 800;
        letter-spacing: 0;
        transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    }

    .stButton > button:hover {
        background: #39d2ca;
        border-color: rgba(39, 195, 189, 0.9);
        color: #06111f;
        transform: translateY(-1px);
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid rgba(110, 168, 254, 0.26);
        background: rgba(37, 83, 139, 0.28);
        color: var(--text);
    }

    .section-spacer {
        height: 0.65rem;
    }

    .footer {
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.86rem;
        margin-top: 2rem;
        padding-top: 1rem;
        text-align: center;
    }

    div[data-testid="stDecoration"] {
        display: none;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 1.2rem 1rem 2rem;
        }

        .hero {
            padding: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def init_llm_model():
    """Load the LLM once into memory and cache it."""
    load_trained_model()


init_llm_model()


def load_reward_curve():
    try:
        if os.path.exists("./logs/reward_curve.json"):
            with open("./logs/reward_curve.json", "r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return []


def random_agent(state) -> GridAction:
    import random

    ren = random.uniform(0, 0.8)
    foss = random.uniform(0, 1.0 - ren)
    bat = random.uniform(-1, 1)
    return GridAction(renewable_ratio=ren, fossil_ratio=foss, battery_action=bat)


def trained_agent(state) -> GridAction:
    return local_llm_agent(state, st.session_state.current_task)


def init_session():
    if "env" not in st.session_state:
        st.session_state.env = EcoGridEnv()
        st.session_state.current_task = "medium"
        st.session_state.state = st.session_state.env.reset(task="medium", seed=42)
        st.session_state.history = []
        st.session_state.cumulative_reward = 0.0
        st.session_state.last_action = None


def reset_session(task):
    st.session_state.current_task = task
    st.session_state.env = EcoGridEnv()
    st.session_state.state = st.session_state.env.reset(task=task, seed=42)
    st.session_state.history = []
    st.session_state.cumulative_reward = 0.0
    st.session_state.last_action = None


def step_env(agent_type):
    env = st.session_state.env
    state = st.session_state.state

    if env.is_done:
        return

    if agent_type == "Random":
        action = random_agent(state)
    elif agent_type == "Heuristic":
        action = heuristic_agent(state, st.session_state.current_task)
    else:
        action = trained_agent(state)

    result = env.step(action)
    st.session_state.state = result.observation
    st.session_state.cumulative_reward += result.reward
    st.session_state.last_action = action

    log_entry = {
        "step": env.current_step,
        "demand": state.demand,
        "reward": result.reward,
        "cost_score": result.info["reward_breakdown"]["cost_score"],
        "carbon_score": result.info["reward_breakdown"]["carbon_score"],
        "stability_score": result.info["reward_breakdown"]["stability_score"],
        "emissions": result.info["carbon_emitted_step"],
    }
    st.session_state.history.append(log_entry)


def base_layout(height, title=None):
    layout = dict(
        height=height,
        margin=dict(l=12, r=12, t=34 if title else 16, b=24),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(color=TEXT, family="Inter, Arial, sans-serif"),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(color=MUTED, size=13), x=0.02)
    return layout


def format_pct(value):
    return f"{value * 100:.0f}%"


def kpi(label, value, note):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


init_session()

with st.sidebar:
    st.title("EcoGrid Controls")
    st.caption("Configure and run the simulation episode.")

    task = st.selectbox("Task difficulty", ["easy", "medium", "hard"], index=1)
    if task != st.session_state.current_task:
        reset_session(task)

    agent = st.radio("Agent policy", ["Random", "Heuristic", "Trained (LoRA)"], index=1)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    if st.button("Step"):
        step_env(agent)

    if st.button("Run Episode"):
        while not st.session_state.env.is_done:
            step_env(agent)

    if st.button("Reset"):
        reset_session(task)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    config = st.session_state.env.get_task_config(st.session_state.current_task)
    st.caption(config["description"])

state = st.session_state.state
config = st.session_state.env.get_task_config(st.session_state.current_task)
episode_length = config["episode_length"]
progress = state.time_step / episode_length if episode_length else 0
history = st.session_state.history
avg_reward = (
    sum(row["reward"] for row in history) / len(history)
    if history
    else 0.0
)
total_emissions = sum(row["emissions"] for row in history) if history else 0.0
budget_used = max(config["carbon_budget"] - state.carbon_budget_remaining, 0.0)
budget_ratio = (
    state.carbon_budget_remaining / config["carbon_budget"]
    if config["carbon_budget"]
    else 0.0
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Sustainable grid reinforcement learning</div>
        <h1>EcoGrid OpenEnv</h1>
        <p>Monitor agent decisions, grid health, carbon budget, and reward quality across a live simulation episode.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi("Episode Progress", f"{state.time_step} / {episode_length}", f"{progress * 100:.0f}% complete")
with k2:
    kpi("Average Reward", f"{avg_reward:.3f}", f"Cumulative {st.session_state.cumulative_reward:.2f}")
with k3:
    kpi("Grid Stability", format_pct(state.grid_stability), f"Battery {format_pct(state.battery_level)}")
with k4:
    kpi("Carbon Remaining", f"{state.carbon_budget_remaining:.0f}", f"{total_emissions:.1f} kgCO2 emitted")

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
col_live, col_reward, col_emissions = st.columns([1, 1, 1])

with col_live:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title"><strong>Live Grid State</strong><span>Telemetry</span></div>',
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Demand", f"{state.demand:.1f} MWh")
        with m2:
            st.metric("Spot Price", f"${state.price_signal:.0f}/MWh")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=state.battery_level * 100,
                number={"suffix": "%", "font": {"color": TEXT, "size": 42}},
                title={"text": "Battery charge", "font": {"size": 13, "color": MUTED}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": GRID},
                    "bar": {"color": PRIMARY, "thickness": 0.22},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 20], "color": "rgba(240, 82, 82, 0.24)"},
                        {"range": [20, 80], "color": "rgba(39, 195, 189, 0.12)"},
                        {"range": [80, 100], "color": "rgba(69, 212, 131, 0.18)"},
                    ],
                },
            )
        )
        fig.update_layout(**base_layout(190))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        fig2 = go.Figure(
            data=[
                go.Bar(name="Demand", x=["Demand"], y=[state.demand], marker_color=DANGER),
                go.Bar(name="Solar", x=["Solar"], y=[state.solar_capacity * 100], marker_color=WARNING),
                go.Bar(name="Wind", x=["Wind"], y=[state.wind_capacity * 100], marker_color=ACCENT),
            ]
        )
        fig2.update_layout(**base_layout(210, "Demand and renewable capacity"))
        fig2.update_layout(barmode="group", showlegend=False, yaxis_title="MWh / capacity %")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

with col_reward:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title"><strong>Agent Performance</strong><span>Reward</span></div>',
            unsafe_allow_html=True,
        )

        if history:
            df = pd.DataFrame(history)

            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=df["reward"],
                    mode="lines",
                    fill="tozeroy",
                    name="Reward",
                    line=dict(color=PRIMARY, width=3),
                    fillcolor="rgba(39, 195, 189, 0.18)",
                )
            )
            fig3.update_layout(**base_layout(210, "Step reward"))
            fig3.update_layout(yaxis_range=[0, 1], xaxis_title="Step", yaxis_title="Reward")
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df["step"], y=df["cost_score"], name="Cost", line=dict(color=WARNING, width=2)))
            fig4.add_trace(go.Scatter(x=df["step"], y=df["carbon_score"], name="Carbon", line=dict(color=SUCCESS, width=2)))
            fig4.add_trace(go.Scatter(x=df["step"], y=df["stability_score"], name="Stability", line=dict(color=ACCENT, width=2)))
            fig4.update_layout(**base_layout(220, "Reward breakdown"))
            fig4.update_layout(
                yaxis_range=[0, 1],
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Press Step or Run Episode in the sidebar to populate performance charts.", icon=None)
            fig_empty = go.Figure()
            fig_empty.add_annotation(
                text="No episode data yet",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color=MUTED, size=18),
            )
            fig_empty.update_layout(**base_layout(430))
            fig_empty.update_xaxes(visible=False)
            fig_empty.update_yaxes(visible=False)
            st.plotly_chart(fig_empty, use_container_width=True, config={"displayModeBar": False})

with col_emissions:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title"><strong>Emissions and Training</strong><span>Carbon</span></div>',
            unsafe_allow_html=True,
        )

        fig5 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=state.carbon_budget_remaining,
                number={"valueformat": ".0f", "font": {"color": TEXT, "size": 42}},
                title={"text": "Carbon budget remaining", "font": {"size": 13, "color": MUTED}},
                gauge={
                    "axis": {"range": [0, config["carbon_budget"]], "tickwidth": 1, "tickcolor": GRID},
                    "bar": {"color": SUCCESS if budget_ratio > 0.2 else DANGER, "thickness": 0.22},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, config["carbon_budget"] * 0.2], "color": "rgba(240, 82, 82, 0.24)"},
                        {
                            "range": [config["carbon_budget"] * 0.2, config["carbon_budget"]],
                            "color": "rgba(69, 212, 131, 0.12)",
                        },
                    ],
                },
            )
        )
        fig5.update_layout(**base_layout(210))
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

        e1, e2 = st.columns(2)
        with e1:
            st.metric("Budget Used", f"{budget_used:.1f} kg")
        with e2:
            st.metric("Mode", st.session_state.current_task.title())

        curve_data = load_reward_curve()
        if curve_data:
            df_curve = pd.DataFrame(curve_data)
            fig6 = go.Figure()
            fig6.add_trace(
                go.Scatter(
                    x=df_curve["step"],
                    y=df_curve["reward"],
                    mode="lines",
                    line=dict(color=PRIMARY, width=3),
                )
            )
            fig6.update_layout(**base_layout(200, "Training reward curve"))
            fig6.update_layout(xaxis_title="Training steps", yaxis_title="Average reward")
            st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Training reward curve")
            st.image("docs/reward_curve.png", caption="Submitted GRPO reward curve", use_column_width=True)

st.markdown('<div class="footer">EcoGrid OpenEnv - Hackathon finale submission</div>', unsafe_allow_html=True)
