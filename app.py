"""
EcoGrid-OpenEnv — Streamlit Dashboard

A 3-panel interactive dashboard for visualizing the RL environment, 
demonstrating the difference between random, heuristic, and trained agents.
Designed for HuggingFace Spaces.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os

from env.environment import EcoGridEnv
from models.schemas import GridAction
from baseline import heuristic_agent, local_llm_agent, load_trained_model

# Use wide mode
st.set_page_config(
    page_title="EcoGrid RL Environment",
    page_icon="⚡",
    layout="wide",
)

CHART_FONT = "#d7dee9"
MUTED_FONT = "#94a3b8"
GRID_COLOR = "rgba(148, 163, 184, 0.14)"
PANEL_BG = "rgba(15, 23, 42, 0)"


def style_chart(fig: go.Figure, height: int, top: int = 30) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=top, b=18),
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=CHART_FONT, family="Inter, Segoe UI, sans-serif", size=12),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    return fig


def style_gauge(fig: go.Figure, height: int = 205) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=34, r=34, t=38, b=8),
        paper_bgcolor=PANEL_BG,
        font=dict(color=CHART_FONT, family="Inter, Segoe UI, sans-serif", size=12),
    )
    return fig

@st.cache_resource
def init_llm_model():
    """Load the LLM once into memory and cache it."""
    load_trained_model()

init_llm_model()

def load_reward_curve():
    try:
        if os.path.exists("./logs/reward_curve.json"):
            with open("./logs/reward_curve.json", "r") as f:
                return json.load(f)
    except:
        pass
    return []

def random_agent(state) -> GridAction:
    import random
    ren = random.uniform(0, 0.8)
    foss = random.uniform(0, 1.0 - ren)
    bat = random.uniform(-1, 1)
    return GridAction(renewable_ratio=ren, fossil_ratio=foss, battery_action=bat)

def trained_agent(state) -> GridAction:
    # Uses the real LLM inference if LoRA is available!
    return local_llm_agent(state, st.session_state.current_task)

def init_session():
    if "env" not in st.session_state:
        st.session_state.env = EcoGridEnv()
        st.session_state.current_task = "medium"
        st.session_state.state = st.session_state.env.reset(task="medium", seed=42)
        st.session_state.history = []
        st.session_state.cumulative_reward = 0.0

def step_env(agent_type):
    env = st.session_state.env
    state = st.session_state.state
    
    if env.is_done:
        return
        
    if agent_type == "Random":
        action = random_agent(state)
    elif agent_type == "Heuristic":
        action = heuristic_agent(state, st.session_state.current_task)
    else: # Trained
        action = trained_agent(state)
        
    result = env.step(action)
    st.session_state.state = result.observation
    st.session_state.cumulative_reward += result.reward
    
    # Save history for plotting
    log_entry = {
        "step": env.current_step,
        "demand": state.demand,
        "reward": result.reward,
        "cost_score": result.info["reward_breakdown"]["cost_score"],
        "carbon_score": result.info["reward_breakdown"]["carbon_score"],
        "stability_score": result.info["reward_breakdown"]["stability_score"],
        "emissions": result.info["carbon_emitted_step"]
    }
    st.session_state.history.append(log_entry)

init_session()

st.markdown("""
    <style>
    :root {
        --surface: #111827;
        --surface-strong: #0f172a;
        --surface-soft: rgba(30, 41, 59, 0.72);
        --line: rgba(148, 163, 184, 0.18);
        --text: #e5edf7;
        --muted: #94a3b8;
        --accent: #20c7b3;
        --accent-2: #58a6ff;
        --warning: #f5c84c;
        --danger: #f05d5e;
    }

    .stApp {
        background:
            radial-gradient(circle at 18% 8%, rgba(32, 199, 179, 0.12), transparent 28rem),
            linear-gradient(135deg, #0b1120 0%, #111827 52%, #0f172a 100%);
        color: var(--text);
        font-family: Inter, "Segoe UI", sans-serif;
    }

    .block-container {
        max-width: 1560px;
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid var(--line);
        min-width: 18rem !important;
        max-width: 18.75rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {
        color: var(--text) !important;
        font-weight: 650;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--line);
        margin: 1.35rem 0;
    }

    .sidebar-brand {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        margin: 0.35rem 0 1.25rem;
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 2.45rem;
        height: 2.45rem;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        color: #06111f;
        font-weight: 900;
        box-shadow: 0 12px 30px rgba(32, 199, 179, 0.22);
    }

    .sidebar-brand h2 {
        margin: 0;
        font-size: 1.08rem;
        letter-spacing: 0;
    }

    .sidebar-brand span {
        color: var(--muted);
        font-size: 0.78rem;
    }

    .hero {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
        background:
            linear-gradient(135deg, rgba(32, 199, 179, 0.12), rgba(88, 166, 255, 0.06)),
            rgba(15, 23, 42, 0.76);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
    }

    .hero-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .hero h1 {
        margin: 0;
        color: var(--text);
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.05;
        letter-spacing: 0;
    }

    .hero h1 span {
        color: var(--accent);
    }

    .hero p {
        margin: 0.65rem 0 0;
        color: var(--muted);
        max-width: 760px;
        font-size: 1rem;
    }

    .run-pill {
        border: 1px solid rgba(32, 199, 179, 0.38);
        background: rgba(32, 199, 179, 0.10);
        color: #9ff5e8;
        border-radius: 999px;
        padding: 0.45rem 0.78rem;
        font-weight: 800;
        font-size: 0.78rem;
        white-space: nowrap;
    }

    .kpi-card {
        min-height: 106px;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.72);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .kpi-value {
        color: var(--text);
        font-size: 1.8rem;
        font-weight: 850;
        margin-top: 0.35rem;
        line-height: 1.1;
    }

    .kpi-note {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    .panel-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        color: var(--text);
        font-size: 1rem;
        font-weight: 800;
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.85rem;
        margin-bottom: 1rem;
    }

    .panel-title small {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
    }

    .section-caption {
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 750;
        margin: 0.6rem 0 0.15rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.86);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }

    div[data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-size: 1.75rem !important;
        font-weight: 850 !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 2.7rem;
        border: 0;
        border-radius: 8px;
        background: linear-gradient(135deg, #20c7b3, #248bd6);
        color: #f8fbff;
        font-weight: 800;
        box-shadow: 0 12px 28px rgba(32, 199, 179, 0.18);
        transition: transform 0.16s ease, filter 0.16s ease, box-shadow 0.16s ease;
    }

    .stButton > button:hover {
        color: #ffffff;
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(32, 199, 179, 0.25);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 8px !important;
        background: rgba(15, 23, 42, 0.66) !important;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.18);
    }

    div[data-testid="column"] {
        min-width: 0 !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid rgba(32, 199, 179, 0.22);
        background: rgba(32, 199, 179, 0.10);
    }

    div[data-testid="stDecoration"] {
        display: none;
    }

    .footer {
        text-align: center;
        padding: 1.6rem 0 0.25rem;
        color: #7f8da3;
        font-size: 0.82rem;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 17rem !important;
            width: auto !important;
        }
        .kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .hero {
            padding: 1rem;
        }
        .kpi-card {
            min-height: 92px;
        }
        .kpi-value {
            font-size: 1.45rem;
        }
    }

    @media (max-width: 560px) {
        .kpi-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-mark">E</div>
        <div>
            <h2>EcoGrid Control</h2>
            <span>Simulation command center</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    task = st.selectbox("Task difficulty", ["easy", "medium", "hard"], index=1)
    if task != st.session_state.current_task:
        st.session_state.current_task = task
        st.session_state.env = EcoGridEnv()
        st.session_state.state = st.session_state.env.reset(task=task, seed=42)
        st.session_state.history = []
        st.session_state.cumulative_reward = 0.0
        
    agent = st.radio("Agent policy", ["Random", "Heuristic", "Trained (LoRA)"], index=1)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Step", use_container_width=True):
            step_env(agent)
    with col2:
        if st.button("Run", use_container_width=True):
            while not st.session_state.env.is_done:
                step_env(agent)
                
    if st.button("Reset episode", use_container_width=True):
        st.session_state.env = EcoGridEnv()
        st.session_state.state = st.session_state.env.reset(task=task, seed=42)
        st.session_state.history = []
        st.session_state.cumulative_reward = 0.0

    st.markdown("---")
    st.markdown("**Model status**")
    policy_path = os.getenv("LORA_ADAPTER_DIR", "/app/lora_adapter")
    if os.path.exists(os.path.join(policy_path, "adapter_config.json")):
        try:
            with open(os.path.join(policy_path, "adapter_config.json")) as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model", "nomic-ai/nomic-embed-text-v1").split("/")[-1]
            st.success(f"Loaded: {base_model}", icon="✅")
            st.markdown(f"Policy: **{base_model} + LoRA**")
        except Exception:
            st.success("Loaded: custom LoRA")
    else:
        st.success("Loaded: default model")
        
    st.markdown("---")
    
    if "training_status" not in st.session_state:
        st.session_state.training_status = "idle"
    
    if st.button("Train GRPO", use_container_width=True):
        st.session_state.training_status = "running"
        def run_training_stub():
            import subprocess, sys, os
            subprocess.Popen([
                sys.executable, "train_unsloth.py", 
            ], cwd=os.getcwd())
        import threading
        t = threading.Thread(target=run_training_stub, daemon=True)
        t.start()
        st.session_state.training_status = "running"
        st.rerun()
        
    if st.button("Save model", use_container_width=True):
        st.session_state.lora_adapter_path = "lora_adapter/best"
        st.success("Model saved (simulated)")
        
    st.markdown("---")
    if st.button("Auto-run episode", use_container_width=True):
        st.session_state.auto_run = not st.session_state.get("auto_run", False)

# Auto-run loop execution
if st.session_state.get("auto_run", False):
    import time
    if "last_run" not in st.session_state:
        st.session_state.last_run = 0
    if time.time() - st.session_state.last_run > 1:
        while not st.session_state.env.is_done:
            step_env(agent)
        st.session_state.last_run = time.time()
        st.rerun()

# ── Main UI ──
state = st.session_state.state
current_config = st.session_state.env.get_task_config(st.session_state.current_task)
episode_length = current_config["episode_length"]
carbon_budget = current_config["carbon_budget"]
progress_pct = min(100, (state.time_step / max(episode_length, 1)) * 100)
carbon_used = carbon_budget - state.carbon_budget_remaining
avg_reward = (
    pd.DataFrame(st.session_state.history)["reward"].mean()
    if st.session_state.history
    else 0.0
)

st.markdown("""
<div class="hero">
    <div class="hero-row">
        <div>
            <div class="eyebrow">Sustainable grid reinforcement learning</div>
            <h1>EcoGrid <span>OpenEnv</span></h1>
            <p>Professional simulation dashboard for balancing demand, renewable variability, battery storage, cost, and carbon constraints.</p>
        </div>
        <div class="run-pill">Live episode monitor</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Episode progress</div>
        <div class="kpi-value">{state.time_step}/{episode_length}</div>
        <div class="kpi-note">{progress_pct:.0f}% complete</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Grid stability</div>
        <div class="kpi-value">{state.grid_stability * 100:.0f}%</div>
        <div class="kpi-note">frequency health indicator</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Carbon used</div>
        <div class="kpi-value">{carbon_used:.0f} kg</div>
        <div class="kpi-note">{state.carbon_budget_remaining:.0f} kg remaining</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Average reward</div>
        <div class="kpi-value">{avg_reward:.3f}</div>
        <div class="kpi-note">{agent} policy</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_live, col_reward, col_emissions = st.columns([1, 1.05, 1])

# Panel 1: Live Grid State
with col_live:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Live grid state <small>current timestep</small></div>', unsafe_allow_html=True)
        state = st.session_state.state
        
        st.metric("Timestep", f"{state.time_step} / {episode_length}")
        
        # Battery Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = state.battery_level * 100,
            title = {'text': "Battery Level (%)", 'font': {'size': 13, 'color': '#a0aec0'}},
            gauge = {
                'axis': {'range': [0, 100], 'visible': False},
                'threshold': {'line': {'color': "#e5edf7", 'width': 2}, 'thickness': 0.72, 'value': state.battery_level * 100},
                'bar': {'color': "#20c7b3", 'thickness': 0.24},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(240, 93, 94, 0.28)'},
                    {'range': [20, 80], 'color': 'rgba(32, 199, 179, 0.12)'},
                    {'range': [80, 100], 'color': 'rgba(81, 207, 141, 0.24)'}
                ]
            }
        ))
        style_gauge(fig)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Capacity Bars
        fig2 = go.Figure(data=[
            go.Bar(name='Demand', x=['Demand'], y=[state.demand], marker_color='#f05d5e', marker_line_width=0, opacity=0.92),
            go.Bar(name='Solar', x=['Solar'], y=[state.solar_capacity * 100], marker_color='#f5c84c', marker_line_width=0, opacity=0.92),
            go.Bar(name='Wind', x=['Wind'], y=[state.wind_capacity * 100], marker_color='#58a6ff', marker_line_width=0, opacity=0.92)
        ])
        style_chart(fig2, 210, 12)
        fig2.update_layout(barmode='group', showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})


# Panel 2: Reward Over Time
with col_reward:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Agent performance <small>reward signals</small></div>', unsafe_allow_html=True)
        
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            
            # Current Episode Reward
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df['step'], y=df['reward'], mode='lines', fill='tozeroy', name='Reward', line=dict(color='#b38cff', width=3), fillcolor='rgba(179, 140, 255, 0.18)'))
            style_chart(fig3, 190, 34)
            fig3.update_layout(title=dict(text="Step reward", font=dict(color=MUTED_FONT, size=13)), xaxis_title="Step", yaxis_title="Reward (0-1)")
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
            
            # Breakdown
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df['step'], y=df['cost_score'], name='Cost', line=dict(dash='dot', color='#f5c84c', width=2.4)))
            fig4.add_trace(go.Scatter(x=df['step'], y=df['carbon_score'], name='Carbon', line=dict(dash='dash', color='#51cf8d', width=2.4)))
            fig4.add_trace(go.Scatter(x=df['step'], y=df['stability_score'], name='Stability', line=dict(color='#58a6ff', width=2.4)))
            style_chart(fig4, 215, 34)
            fig4.update_layout(title=dict(text="Reward breakdown", font=dict(color=MUTED_FONT, size=13)), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=CHART_FONT)))
            st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Press 'Step' or 'Run Episode' in the sidebar to see performance charts.", icon=None)
            for _ in range(12): st.empty() # padding to match height

# Panel 3: Emissions & Training
with col_emissions:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Emissions and training <small>carbon guardrail</small></div>', unsafe_allow_html=True)
        
        # Carbon Budget Gauge
        max_budget = carbon_budget
        current_budget = state.carbon_budget_remaining
        gauge_min = min(0, current_budget)
        
        fig5 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_budget,
            title = {'text': "Carbon Budget (kgCO2)", 'font': {'size': 13, 'color': '#a0aec0'}},
            number = {'valueformat': ".0f", 'font': {'color': '#e2e8f0'}},
            gauge = {
                'axis': {'range': [gauge_min, max_budget], 'visible': False},
                'bar': {'color': "#51cf8d" if current_budget > max_budget * 0.2 else "#f05d5e", 'thickness': 0.24},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [gauge_min, max_budget * 0.2], 'color': "rgba(240, 93, 94, 0.22)"}
                ]
            }
        ))
        style_gauge(fig5)
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
        
        # RL Training Curve
        st.markdown("<div class='section-caption'>GRPO training progress</div>", unsafe_allow_html=True)
        curve_data = load_reward_curve()
        if curve_data:
            df_curve = pd.DataFrame(curve_data)
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=df_curve['step'], y=df_curve['reward'], mode='lines', line=dict(color='#20c7b3', width=3)))
            style_chart(fig6, 190, 14)
            fig6.update_layout(xaxis_title="Training steps", yaxis_title="Avg reward")
            st.plotly_chart(fig6, use_container_width=True, config={'displayModeBar': False})
        else:
            st.image("docs/reward_curve.png", caption="GRPO reward curve from the submitted training run")

st.markdown("""
<div class="footer">
    EcoGrid OpenEnv — Hackathon Finale Submission
</div>
""", unsafe_allow_html=True)

