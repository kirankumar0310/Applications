import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="CLT Visualizer")
st.title("📊 Central Limit Theorem Demo")

# Sidebar inputs
st.sidebar.header("Parameters")
dist_choice = st.sidebar.selectbox(
    "Distribution", 
    ["Exponential", "Uniform", "Bimodal"]
)
n = st.sidebar.slider("Sample Size (n)", min_value=1, max_value=100, value=30)
num_sims = st.sidebar.slider("Simulations", min_value=500, max_value=5000, value=2000, step=500)

np.random.seed(42)

# Generate data
if dist_choice == "Exponential":
    scale = 2.0
    pop = np.random.exponential(scale, 40_000)
    samples = np.random.exponential(scale, (num_sims, n))
    pop_mean, pop_std = scale, scale
    xlim = [0, 10]
elif dist_choice == "Uniform":
    low, high = 0.0, 10.0
    pop = np.random.uniform(low, high, 40_000)
    samples = np.random.uniform(low, high, (num_sims, n))
    pop_mean = (low + high) / 2.0
    pop_std = np.sqrt(((high - low) ** 2) / 12.0)
    xlim = [-1, 11]
else:
    pop = np.concatenate([np.random.normal(2, 0.6, 20_000), np.random.normal(8, 0.6, 20_000)])
    comp = np.random.binomial(1, 0.5, size=(num_sims, n))
    samples = np.where(comp == 1, np.random.normal(2, 0.6, (num_sims, n)), np.random.normal(8, 0.6, (num_sims, n)))
    pop_mean = 5.0
    pop_std = float(np.std(pop))
    xlim = [0, 10]

sample_means = np.mean(samples, axis=1)
se = pop_std / np.sqrt(n)

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Population μ", f"{pop_mean:.2f}")
col2.metric("Mean of Sample Means", f"{np.mean(sample_means):.2f}")
col3.metric("Theoretical SE (σ/√n)", f"{se:.3f}")

# Plots
fig = make_subplots(rows=1, cols=2, subplot_titles=["Population Distribution", f"Sample Means Distribution (n={n})"])
fig.add_trace(go.Histogram(x=pop, histnorm='probability density', marker_color='#f39c12', opacity=0.7, name="Pop"), row=1, col=1)
fig.add_trace(go.Histogram(x=sample_means, histnorm='probability density', marker_color='#2980b9', opacity=0.6, name="Means"), row=1, col=2)

# Normal curve overlay
x_curve = np.linspace(pop_mean - 4*se, pop_mean + 4*se, 200)
fig.add_trace(go.Scatter(x=x_curve, y=norm.pdf(x_curve, pop_mean, se), mode='lines', name='CLT Fit', line=dict(color='crimson', width=2)), row=1, col=2)

fig.update_xaxes(range=xlim, row=1, col=1)
fig.update_layout(height=450, bargap=0.05, margin=dict(t=40, b=20, l=20, r=20))
st.plotly_chart(fig, use_container_width=True)