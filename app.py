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
st.set_page_config(page_title="EcoGrid RL Environment", layout="wide")

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

# ── Sidebar ──
with st.sidebar:
    st.title("⚡ EcoGrid Config")
    
    task = st.selectbox("Select Task Difficulty", ["easy", "medium", "hard"], index=1)
    if task != st.session_state.current_task:
        st.session_state.current_task = task
        st.session_state.env = EcoGridEnv()
        st.session_state.state = st.session_state.env.reset(task=task, seed=42)
        st.session_state.history = []
        st.session_state.cumulative_reward = 0.0
        
    agent = st.radio("Select Agent", ["Random", "Heuristic", "Trained (LoRA)"], index=1)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Step"):
            step_env(agent)
    with col2:
        if st.button("⏭ Run Episode"):
            while not st.session_state.env.is_done:
                step_env(agent)
                
    if st.button("🔄 Reset"):
        st.session_state.env = EcoGridEnv()
        st.session_state.state = st.session_state.env.reset(task=task, seed=42)
        st.session_state.history = []
        st.session_state.cumulative_reward = 0.0

    st.markdown("---")
    st.markdown("**Agent policy:**")
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
    
    if st.button("🚀 Train GRPO"):
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
        
    if st.button("💾 Save Model"):
        st.session_state.lora_adapter_path = "lora_adapter/best"
        st.success("Model saved (simulated)")
        
    st.markdown("---")
    if st.button("🔄 Auto-run"):
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
st.markdown("""
<div class="main-header">
    <h1>🌍 EcoGrid <span class="highlight">OpenEnv</span></h1>
    <p>Production-Grade RL Environment for Sustainable Energy Grid Management</p>
</div>
""", unsafe_allow_html=True)

col_live, col_reward, col_emissions = st.columns(3)

# Panel 1: Live Grid State
with col_live:
    with st.container(border=True):
        st.markdown('<div class="panel-title">📡 Live Grid State</div>', unsafe_allow_html=True)
        state = st.session_state.state
        
        st.metric("Timestep", f"{state.time_step} / {st.session_state.env.get_task_config(st.session_state.current_task)['episode_length']}")
        
        # Battery Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = state.battery_level * 100,
            title = {'text': "Battery Level (%)", 'font': {'size': 13, 'color': '#a0aec0'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4a5568"},
                'bar': {'color': "#38b2ac"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(229, 62, 62, 0.2)'},
                    {'range': [20, 80], 'color': 'rgba(56, 178, 172, 0.1)'},
                    {'range': [80, 100], 'color': 'rgba(72, 187, 120, 0.2)'}
                ]
            }
        ))
        fig.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': '#e2e8f0'})
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Capacity Bars
        fig2 = go.Figure(data=[
            go.Bar(name='Demand', x=['Demand'], y=[state.demand], marker_color='#e53e3e', marker_line_width=0, opacity=0.9),
            go.Bar(name='Solar', x=['Solar'], y=[state.solar_capacity * 100], marker_color='#ecc94b', marker_line_width=0, opacity=0.9),
            go.Bar(name='Wind', x=['Wind'], y=[state.wind_capacity * 100], marker_color='#4299e1', marker_line_width=0, opacity=0.9)
        ])
        fig2.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=20), barmode='group', showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(gridcolor="#2d3748"))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})


# Panel 2: Reward Over Time
with col_reward:
    with st.container(border=True):
        st.markdown('<div class="panel-title">📈 Agent Performance</div>', unsafe_allow_html=True)
        
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            
            # Current Episode Reward
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df['step'], y=df['reward'], mode='lines', fill='tozeroy', name='Reward', line=dict(color='#9f7aea', width=3), fillcolor='rgba(159, 122, 234, 0.2)'))
            fig3.update_layout(title=dict(text="Step Reward", font=dict(color="#a0aec0", size=13)), height=180, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Step", yaxis_title="Reward (0-1)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"), font={'color': '#e2e8f0'})
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
            
            # Breakdown
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df['step'], y=df['cost_score'], name='Cost', line=dict(dash='dot', color='#f6e05e', width=2)))
            fig4.add_trace(go.Scatter(x=df['step'], y=df['carbon_score'], name='Carbon', line=dict(dash='dash', color='#68d391', width=2)))
            fig4.add_trace(go.Scatter(x=df['step'], y=df['stability_score'], name='Stability', line=dict(color='#63b3ed', width=2)))
            fig4.update_layout(title=dict(text="Reward Breakdown", font=dict(color="#a0aec0", size=13)), height=200, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0")), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"), font={'color': '#e2e8f0'})
            st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Press 'Step' or 'Run Episode' in the sidebar to see performance charts.", icon=None)
            for _ in range(12): st.empty() # padding to match height

# Panel 3: Emissions & Training
with col_emissions:
    with st.container(border=True):
        st.markdown('<div class="panel-title">🌍 Emissions & Training</div>', unsafe_allow_html=True)
        
        # Carbon Budget Gauge
        max_budget = st.session_state.env.get_task_config(st.session_state.current_task)['carbon_budget']
        current_budget = state.carbon_budget_remaining
        
        fig5 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_budget,
            title = {'text': "Carbon Budget (kgCO2)", 'font': {'size': 13, 'color': '#a0aec0'}},
            number = {'valueformat': ".0f", 'font': {'color': '#e2e8f0'}},
            gauge = {
                'axis': {'range': [0, max_budget], 'tickwidth': 1, 'tickcolor': "#4a5568"},
                'bar': {'color': "#48bb78" if current_budget > max_budget * 0.2 else "#e53e3e"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, max_budget * 0.2], 'color': "rgba(229, 62, 62, 0.2)"}
                ]
            }
        ))
        fig5.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': '#e2e8f0'})
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
        
        # RL Training Curve
        st.markdown("<div style='font-size: 13px; color: #a0aec0; margin-top: 10px; margin-bottom: -10px;'>🧠 GRPO Training Progress (Unsloth)</div>", unsafe_allow_html=True)
        curve_data = load_reward_curve()
        if curve_data:
            df_curve = pd.DataFrame(curve_data)
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=df_curve['step'], y=df_curve['reward'], mode='lines', line=dict(color='#38b2ac', width=3)))
            fig6.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=20), xaxis_title="Training Steps", yaxis_title="Avg Reward", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"), font={'color': '#e2e8f0'})
            st.plotly_chart(fig6, use_container_width=True, config={'displayModeBar': False})
        else:
            st.image("docs/reward_curve.png", caption="GRPO reward curve from the submitted training run")

st.markdown("""
<div class="footer">
    EcoGrid OpenEnv — Hackathon Finale Submission
</div>
""", unsafe_allow_html=True)

# Global styling tweaks for clean padding and professional glassmorphism look
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1a202c 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38b2ac, #4299e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header .highlight {
        color: #e2e8f0;
        -webkit-text-fill-color: #e2e8f0;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        color: #a0aec0;
        font-size: 1.1rem;
    }
    
    /* Panel Containers (Glassmorphism) */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background: rgba(26, 32, 44, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2) !important;
    }

    /* Panel Titles */
    .panel-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.5rem;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #38b2ac !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a202c !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stButton>button {
        background: linear-gradient(135deg, #38b2ac 0%, #319795 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4fd1c5 0%, #38b2ac 100%);
        box-shadow: 0 4px 12px rgba(56, 178, 172, 0.3);
        transform: translateY(-1px);
    }
    
    /* Hide Decorations */
    div[data-testid="stDecoration"] {
        display: none;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #718096;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

