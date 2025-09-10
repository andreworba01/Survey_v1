# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from urllib.request import urlopen

# ================================
# 1) Manual scale & thresholds
# ================================
ORDER = ["Very Low", "Low", "Mid-Low", "Mid", "High"]

THRESHOLDS = {
    "p10": 1.0000, "p35": 1.2533, "p50": 1.3740, "p70": 1.5380,
    "p90": 1.8686, "p95": 2.2911, "p99": 2.5193
}

def categorize(score, t=THRESHOLDS):
    """Manual 5-band classifier."""
    if score <= t["p35"]:
        return "Very Low"
    elif score <= t["p70"]:
        return "Low"
    elif score <= t["p90"]:
        return "Mid-Low"
    elif score <= t["p99"]:
        return "Mid"
    else:
        return "High"

# Weights for Q73..Q80 (in this order)
W = np.array([0.05, 0.07, 0.064, 0.16, 0.228, 0.153, 0.138, 0.137])

# ================================
# 2) Load data
# ================================
df_scores = pd.read_csv("Exposure_with_Quartiles.csv")
df_map = pd.read_csv("Nebraska_County_Simulated_Scores.csv")

# pick the score column to show in the histogram
SCORE_COL = "Weighted_Score" if "Weighted_Score" in df_scores.columns else (
    "Total_Exposure_Score" if "Total_Exposure_Score" in df_scores.columns else "Exposure_Score"
)

# ================================
# 3) Sidebar
# ================================
page = st.sidebar.radio("📑 Select View", ["📊 Individual Risk (Manual Scale)", "🗺️ County Risk Map"])

# ================================
# 4) PAGE 1 — Individual
# ================================
if page == "📊 Individual Risk (Manual Scale)":
    st.title("🧪 Microplastic Exposure Risk (Manual Scale)")
    st.caption(
        "Weighted score = 0.05·Q73 + 0.07·Q74 + 0.064·Q75 + 0.16·Q76 + "
        "0.228·Q77 + 0.153·Q78 + 0.138·Q79 + 0.137·Q80"
    )
    st.markdown(
        f"**Percentile thresholds** used for categories: "
        f"p35={THRESHOLDS['p35']:.4f}, p70={THRESHOLDS['p70']:.4f}, "
        f"p90={THRESHOLDS['p90']:.4f}, p99={THRESHOLDS['p99']:.4f}."
    )

    # ---- Questions (same text, manual mapping) ----
    q1 = st.selectbox("73. How many ounces of bottled water do you drink each day, on average?",
        ["Over 50.8 ounces", "34-50.7 ounces", "17-33.9 ounces", "1-16.9 ounces", "Less than 1 ounce"])
    q2 = st.selectbox("74. How many ounces of water do you typically drink each day from refillable plastic water bottles?",
        ["Over 96 ounces", "65-96 ounces", "33-64 ounces", "1-32 ounces", "Less than 1 ounce"])
    q3 = st.selectbox("75. How many ounces of other beverages (juice, soda, sports drinks, etc.) do you typically drink each day from plastic packaging?",
        ["Over 40 ounces", "20-40 ounces", "1-19 ounces", "Less than 1 ounce"])
    q4 = st.selectbox("76. How often, on average each day, do you use disposable cups for hot drinks (e.g., tea, coffee)?",
        ["5 or more times per day", "3-4 times per day", "1-2 times per day", "Less than 1 time per day", "Never"])
    q5 = st.selectbox("77. How often, on average each day, do you use a microwave to heat water or beverages in a disposable cup?",
        ["5 or more times per day", "3-4 times per day", "1-2 times per day", "Less than 1 time per day", "Never"])
    q6 = st.selectbox("78. During the summer, how often each week do you leave bottled water in a car or under direct sunlight for over 30 minutes?",
        ["5 or more days per week", "3-4 days per week", "1-2 days per week", "Never"])
    q7 = st.selectbox("79. During the summer, how often each week do you leave refillable plastic water bottles in a car or under direct sunlight?",
        ["5 or more days per week", "3-4 days per week", "1-2 days per week", "Never"])
    q8 = st.selectbox("80. During the summer, how often each week do you leave other beverages with plastic packaging in a car or under direct sunlight?",
        ["5 or more days per week", "3-4 days per week", "1-2 days per week", "Never"])

    # ---- Map answers to numeric (higher = riskier) ----
    map_5pt = {
        "Over 50.8 ounces": 5, "34-50.7 ounces": 4, "17-33.9 ounces": 3, "1-16.9 ounces": 2, "Less than 1 ounce": 1,
        "Over 96 ounces": 5, "65-96 ounces": 4, "33-64 ounces": 3, "1-32 ounces": 2, "Less than 1 ounce": 1,
        "5 or more times per day": 5, "3-4 times per day": 4, "1-2 times per day": 3, "Less than 1 time per day": 2, "Never": 1
    }
    map_4pt = {
        "Over 40 ounces": 4, "20-40 ounces": 3, "1-19 ounces": 2, "Less than 1 ounce": 1,
        "5 or more days per week": 4, "3-4 days per week": 3, "1-2 days per week": 2, "Never": 1
    }

    # Map raw scores
    responses = np.array([
        map_5pt[q1], map_5pt[q2], map_4pt[q3], map_5pt[q4],
        map_5pt[q5], map_4pt[q6], map_4pt[q7], map_4pt[q8]
    ], dtype=float)
    
    # Normalize (standardize to 0–1 range)
    responses_std = np.array([
        (map_5pt[q1] - 1) / 4,
        (map_5pt[q2] - 1) / 4,
        (map_4pt[q3] - 1) / 3,
        (map_5pt[q4] - 1) / 4,
        (map_5pt[q5] - 1) / 4,
        (map_4pt[q6] - 1) / 3,
        (map_4pt[q7] - 1) / 3,
        (map_4pt[q8] - 1) / 3
    ], dtype=float)
    
    
    # Compute standardized weighted score
    if st.button("🔍 Compute Weighted Score & Category"):
        user_score = float(np.dot(responses_std, W))  # standardized
        user_cat = categorize(user_score)
    
        st.success(f"🧮 Standardized weighted score: **{user_score:.3f}**  →  **{user_cat}**")
        
        # --- Distribution with user's score ---
        hist_x = df_scores[SCORE_COL].dropna()
        # histogram trace
        hist_trace = go.Histogram(
            x=hist_x, nbinsx=30,
            marker=dict(color="#6ce5e8", line=dict(color="black", width=1)),
            opacity=0.7, name="Population"
        )
        # vertical line at user's score
        ymax = np.histogram(hist_x, bins=30)[0].max()
        line_trace = go.Scatter(
            x=[user_score, user_score], y=[0, ymax],
            mode="lines+text",
            name=f"Your score: {user_score:.3f}",
            line=dict(color="crimson", width=3, dash="dash"),
            text=[f"Your score: {user_score:.3f}"], textposition="top right"
        )
        fig = go.Figure([hist_trace, line_trace])
        fig.update_layout(
            title="📊 Distribution of Weighted Scores",
            xaxis_title="Weighted Score", yaxis_title="Count",
            template="plotly_white", hovermode="x", legend=dict(x=0.7, y=0.95)
        )
        st.plotly_chart(fig, use_container_width=True)

        # percentile vs population
        pct = (hist_x < user_score).mean() * 100
        st.markdown(f"🔢 Your score is higher than **{pct:.1f}%** of the population in the dataset.")

# ================================
# 5) PAGE 2 — County Map
# ================================
elif page == "🗺️ County Risk Map":
    st.title("🗺️ Nebraska County — Mean Weighted Score (Manual Scale)")

    # Risk level per county using the same manual thresholds
    def assign_risk(score):
        return categorize(score, THRESHOLDS)

    df_map = df_map.copy()
    df_map["fips"] = df_map["fips"].astype(str).str.zfill(5)
    df_map["Risk_Level"] = df_map["mean_score"].apply(assign_risk)

    # Filter to Nebraska counties (FIPS starts with '31')
    df_ne = df_map[df_map["fips"].str.startswith("31")].copy()

    # GeoJSON for US counties
    with urlopen("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json") as r:
        counties_geo = json.load(r)

    # dynamic color range
    vmin, vmax = df_ne["mean_score"].min(), df_ne["mean_score"].max()

    fig = px.choropleth(
        df_ne,
        geojson=counties_geo,
        locations="fips",
        color="mean_score",
        scope="usa",
        color_continuous_scale="Viridis",
        range_color=(vmin, vmax),
        labels={"mean_score": "Mean Weighted Score"},
        hover_data={"county": True, "Risk_Level": True, "mean_score": ":.3f", "fips": False},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        title="📍 Nebraska Counties — Mean Weighted Score",
        margin=dict(r=0, t=40, l=0, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend text for thresholds
    st.caption(
        f"Manual categories: {ORDER}. "
        f"Cuts — Very Low ≤ {THRESHOLDS['p35']:.3f} < Low ≤ {THRESHOLDS['p70']:.3f} "
        f"< Mid-Low ≤ {THRESHOLDS['p90']:.3f} < Mid ≤ {THRESHOLDS['p99']:.3f} < High."
    )
