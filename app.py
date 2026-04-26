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
import importlib.util

from env.environment import EcoGridEnv
from models.schemas import GridAction
from baseline import heuristic_agent, local_llm_agent, load_trained_model, LORA_DIR, is_lora_valid

# Use wide mode with a custom icon
st.set_page_config(page_title="EcoGrid Dashboard", layout="wide", page_icon="🌍")

@st.cache_data
def init_llm_model():
    """Fast availability check (no heavyweight model load)."""
    required_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    files_ok = is_lora_valid()
    deps_ok = (
        importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("peft") is not None
        and importlib.util.find_spec("torch") is not None
    )
    return files_ok and deps_ok

TRAINED_AVAILABLE = init_llm_model()

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
        st.session_state.trained_runtime_checked = False
        st.session_state.trained_runtime_ready = False
        st.session_state.trained_fallback_used = False

def step_env(agent_type):
    env = st.session_state.env
    state = st.session_state.state
    
    if env.is_done:
        return
        
    if agent_type == "Random Agent":
        action = random_agent(state)
    elif agent_type == "Heuristic Rule-Based":
        action = heuristic_agent(state, st.session_state.current_task)
    else: # Trained
        if not st.session_state.trained_runtime_checked:
            model, _ = load_trained_model()
            st.session_state.trained_runtime_checked = True
            st.session_state.trained_runtime_ready = model is not None

        if st.session_state.trained_runtime_ready:
            action = trained_agent(state)
        else:
            st.session_state.trained_fallback_used = True
            action = heuristic_agent(state, st.session_state.current_task)
        
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

# ─── THEME TOKENS ───
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"
COLOR_GRID = "rgba(255, 255, 255, 0.05)"
COLOR_PRIMARY = "#00f2fe"  # Vibrant teal
COLOR_SECONDARY = "#4facfe" # Soft blue
COLOR_WARN = "#facc15"     # Yellow
COLOR_DANGER = "#ff4b4b"   # Red/Pink
COLOR_SUCCESS = "#00f260"  # Green
COLOR_PURPLE = "#c084fc"   # Accent purple

# ─── SIDEBAR CONTROL PANEL ───
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding-bottom: 20px;'>
            <h2 style='margin: 0; color: #00f2fe;'>⚡ Control Room</h2>
            <p style='color: #94a3b8; font-size: 0.9rem; margin-top: 5px;'>Configure the environment and agent.</p>
        </div>
    """, unsafe_allow_html=True)
    
    task_labels = {"easy": "Easy (No Battery, Flat Demand)", "medium": "Medium (Small Battery, Spikes)", "hard": "Hard (Carbon Cap, High Volatility)"}
    task = st.selectbox(
        "Simulation Difficulty", 
        ["easy", "medium", "hard"], 
        index=1,
        format_func=lambda x: task_labels[x],
        help="Changes the weather volatility, demand curves, and carbon constraints."
    )
    
    if task != st.session_state.current_task:
        st.session_state.current_task = task
        st.session_state.env = EcoGridEnv()
        st.session_state.state = st.session_state.env.reset(task=task, seed=42)
        st.session_state.history = []
        st.session_state.cumulative_reward = 0.0
        st.session_state.trained_runtime_checked = False
        st.session_state.trained_runtime_ready = False
        st.session_state.trained_fallback_used = False
        
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 15px 0;'>", unsafe_allow_html=True)
    
    agent_options = ["Random Agent", "Heuristic Rule-Based"]
    if TRAINED_AVAILABLE:
        agent_options.append("AI Agent (Trained LoRA)")
    agent = st.radio(
        "Active Agent", 
        agent_options, 
        index=1,
        help="Select which intelligence is controlling the grid."
    )
    
    if not TRAINED_AVAILABLE:
        st.warning("AI Agent weights missing (LFS pull required).", icon="⚠️")
    elif st.session_state.trained_fallback_used and not st.session_state.trained_runtime_ready:
        st.error("Incompatible hardware or loading error. Falling back.", icon="🚨")
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 15px 0;'>", unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ Step Once", use_container_width=True):
            step_env(agent)
    with col_btn2:
        if st.button("⏩ Run Full", use_container_width=True):
            while not st.session_state.env.is_done:
                step_env(agent)
                
    if st.button("🔄 Reset Environment", use_container_width=True):
        st.session_state.env = EcoGridEnv()
        st.session_state.state = st.session_state.env.reset(task=task, seed=42)
        st.session_state.history = []
        st.session_state.cumulative_reward = 0.0
        st.session_state.trained_runtime_checked = False
        st.session_state.trained_runtime_ready = False
        st.session_state.trained_fallback_used = False

# ─── MAIN UI HEADER ───
st.markdown("""
<div class="main-header">
    <h1>🌍 EcoGrid <span class="highlight">Intelligence</span></h1>
    <p>AI-Powered Sustainable Energy Grid Management</p>
</div>
""", unsafe_allow_html=True)

if TRAINED_AVAILABLE and st.session_state.trained_runtime_ready:
    st.success("Trained AI Model loaded and active.", icon="🤖")

col_live, col_reward, col_emissions = st.columns(3)

# ─── PANEL 1: LIVE GRID STATE ───
with col_live:
    with st.container(border=True):
        st.markdown('<div class="panel-title">📡 Live Grid State</div>', unsafe_allow_html=True)
        st.markdown('<p class="panel-subtitle">Real-time supply and demand metrics.</p>', unsafe_allow_html=True)
        state = st.session_state.state
        
        # Custom Metric Card
        ep_len = st.session_state.env.get_task_config(st.session_state.current_task)['episode_length']
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-label">Timestep Progress</span>
                <span class="metric-value">{state.time_step} <span style="color:#94a3b8; font-size:1.2rem;">/ {ep_len}</span></span>
            </div>
        """, unsafe_allow_html=True)
        
        # Battery Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = state.battery_level * 100,
            number = {'suffix': "%", 'font': {'color': COLOR_TEXT, 'size': 24}},
            title = {'text': "Battery Level", 'font': {'size': 14, 'color': COLOR_MUTED}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLOR_GRID},
                'bar': {'color': COLOR_PRIMARY, 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 20], 'color': "rgba(255, 75, 75, 0.2)"},
                    {'range': [20, 80], 'color': "rgba(0, 242, 254, 0.1)"},
                    {'range': [80, 100], 'color': "rgba(0, 242, 96, 0.2)"}
                ]
            }
        ))
        fig.update_layout(height=170, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Inter'})
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Capacity Bars
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Demand (MWh)', x=['Demand'], y=[state.demand], marker_color=COLOR_DANGER, opacity=0.8, marker_line_width=0, hoverinfo="y+name"))
        fig2.add_trace(go.Bar(name='Solar (%)', x=['Solar'], y=[state.solar_capacity * 100], marker_color=COLOR_WARN, opacity=0.8, marker_line_width=0, hoverinfo="y+name"))
        fig2.add_trace(go.Bar(name='Wind (%)', x=['Wind'], y=[state.wind_capacity * 100], marker_color=COLOR_SECONDARY, opacity=0.8, marker_line_width=0, hoverinfo="y+name"))
        
        fig2.update_layout(
            height=180, margin=dict(l=10, r=10, t=10, b=20), barmode='group', showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor=COLOR_GRID, showticklabels=False),
            xaxis=dict(tickfont=dict(color=COLOR_TEXT, size=13)),
            font=dict(family='Inter')
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

# ─── PANEL 2: AGENT PERFORMANCE ───
with col_reward:
    with st.container(border=True):
        st.markdown('<div class="panel-title">📈 Agent Performance</div>', unsafe_allow_html=True)
        st.markdown('<p class="panel-subtitle">Multi-objective optimization scoring.</p>', unsafe_allow_html=True)
        
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            
            # Area Chart for Overall Reward
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df['step'], y=df['reward'], mode='lines', fill='tozeroy', 
                name='Total Reward', 
                line=dict(color=COLOR_PRIMARY, width=3), 
                fillcolor='rgba(0, 242, 254, 0.2)',
                hovertemplate="Step %{x}<br>Reward: %{y:.2f}<extra></extra>"
            ))
            fig3.update_layout(
                title=dict(text="Cumulative Step Reward (0-1)", font=dict(color=COLOR_MUTED, size=13)),
                height=180, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor=COLOR_GRID, color=COLOR_MUTED),
                yaxis=dict(gridcolor=COLOR_GRID, color=COLOR_MUTED, range=[0, 1.05]),
                font=dict(family='Inter')
            )
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
            
            # Breakdown Lines
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df['step'], y=df['cost_score'], name='Cost Efficiency', line=dict(color=COLOR_WARN, width=2), hovertemplate="%{y:.2f}"))
            fig4.add_trace(go.Scatter(x=df['step'], y=df['carbon_score'], name='Eco Score', line=dict(color=COLOR_SUCCESS, width=2), hovertemplate="%{y:.2f}"))
            fig4.add_trace(go.Scatter(x=df['step'], y=df['stability_score'], name='Grid Stability', line=dict(color=COLOR_PURPLE, width=2), hovertemplate="%{y:.2f}"))
            
            fig4.update_layout(
                title=dict(text="Objective Breakdown", font=dict(color=COLOR_MUTED, size=13)),
                height=200, margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLOR_TEXT, size=10)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor=COLOR_GRID, color=COLOR_MUTED),
                yaxis=dict(gridcolor=COLOR_GRID, color=COLOR_MUTED, range=[0, 1.05]),
                font=dict(family='Inter'),
                hovermode="x unified"
            )
            st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Run the simulation to view performance graphs.")
            # Blank spacers to maintain identical panel height
            st.markdown("<div style='height: 380px;'></div>", unsafe_allow_html=True)

# ─── PANEL 3: EMISSIONS & TRAINING ───
with col_emissions:
    with st.container(border=True):
        st.markdown('<div class="panel-title">🌱 Constraints & Learning</div>', unsafe_allow_html=True)
        st.markdown('<p class="panel-subtitle">Carbon limits and AI training convergence.</p>', unsafe_allow_html=True)
        
        # Carbon Budget Gauge
        max_budget = st.session_state.env.get_task_config(st.session_state.current_task)['carbon_budget']
        current_budget = state.carbon_budget_remaining
        is_strict = st.session_state.env.get_task_config(st.session_state.current_task)['carbon_strict']
        
        budget_color = COLOR_SUCCESS if current_budget > max_budget * 0.2 else COLOR_DANGER
        if current_budget < 0: budget_color = "#8b0000" # Deep red for failure
        
        fig5 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = max(0, current_budget), # Visual clamp
            number = {'valueformat': ".0f", 'font': {'color': COLOR_TEXT, 'size': 24}},
            title = {'text': f"Carbon Budget (kgCO2) {'⚠️ Strict' if is_strict else ''}", 'font': {'size': 14, 'color': COLOR_MUTED}},
            gauge = {
                'axis': {'range': [0, max_budget], 'tickwidth': 1, 'tickcolor': COLOR_GRID},
                'bar': {'color': budget_color, 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, max_budget * 0.2], 'color': "rgba(255, 75, 75, 0.2)"}
                ]
            }
        ))
        fig5.update_layout(height=170, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(family='Inter'))
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
        
        # Training Curve or Image
        st.markdown("<div style='font-size: 13px; color: #94a3b8; margin-top: 10px; margin-bottom: 5px; font-weight: 500;'>🧠 GRPO Training Convergence</div>", unsafe_allow_html=True)
        curve_data = load_reward_curve()
        if curve_data:
            df_curve = pd.DataFrame(curve_data)
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(
                x=df_curve['step'], y=df_curve['reward'], mode='lines', 
                line=dict(color=COLOR_PRIMARY, width=2),
                fill='tozeroy', fillcolor='rgba(0, 242, 254, 0.1)'
            ))
            fig6.update_layout(
                height=180, margin=dict(l=10, r=10, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor=COLOR_GRID, color=COLOR_MUTED, title="Training Steps"),
                yaxis=dict(gridcolor=COLOR_GRID, color=COLOR_MUTED, title="Avg Reward"),
                font=dict(family='Inter')
            )
            st.plotly_chart(fig6, use_container_width=True, config={'displayModeBar': False})
        else:
            if os.path.exists("docs/reward_curve.png"):
                st.image("docs/reward_curve.png", caption="Historical Training Performance")
            else:
                st.info("No training data available.")

st.markdown("""
<div class="footer">
    Developed by <b>Team DD</b> for the Meta PyTorch Hackathon.
</div>
""", unsafe_allow_html=True)

# ─── GLOBAL STYLING (Glassmorphism & Rich Aesthetics) ───
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 15% 50%, rgba(0, 242, 254, 0.05), transparent 40%),
                    radial-gradient(circle at 85% 30%, rgba(192, 132, 252, 0.05), transparent 40%),
                    linear-gradient(145deg, #090e17 0%, #111827 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Area */
    .main-header {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header .highlight {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 400;
        bit-weight: 400;
    }
    
    /* Panel Containers */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background: rgba(17, 24, 39, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Panel Typography */
    .panel-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .panel-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 0.8rem;
    }
    
    /* Custom Metric Card */
    .metric-card {
        background: rgba(0, 242, 254, 0.03);
        border: 1px solid rgba(0, 242, 254, 0.15);
        padding: 16px 20px;
        border-radius: 14px;
        margin-bottom: 15px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #00f2fe;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 24, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        color: #f8fafc;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.15) 0%, rgba(79, 172, 254, 0.15) 100%);
        border-color: rgba(0, 242, 254, 0.4);
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.2);
        transform: translateY(-1px);
        color: #fff;
    }
    .stButton > button:active {
        transform: translateY(1px);
    }
    
    /* Dropdowns and Inputs */
    div[data-baseweb="select"] > div, input[type="text"], div[data-baseweb="radio"] {
        background-color: rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
    }
    
    /* Hide specific streamlit decorations */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #64748b;
        font-size: 0.95rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }
    .footer b {
        color: #94a3b8;
    }
    </style>
""", unsafe_allow_html=True)
