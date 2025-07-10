
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import plotly.graph_objects as go
import plotly.express as px


# --- Load model and scaler ---

# Load the trained MLP model
model = joblib.load("mlp_model.pkl")

scaler = joblib.load("fitted_scaler.pkl")
class_names = ["High", "High‑Medium", "Medium", "Low‑Medium", "Low"]

# --- Load Data ---
df_scores = pd.read_csv("Exposure_with_Quartiles.csv")
df_map = pd.read_csv("Nebraska_County_Simulated_Scores.csv")

# --- Sidebar Navigation ---
page = st.sidebar.radio("📑 Select View", ["📊 Individual Risk Prediction", "🗺️ County Risk Map"])

# -------------------------
# 🧪 PAGE 1: Prediction
# -------------------------
if page == "📊 Individual Risk Prediction":
    st.title("🧪 Microplastic Exposure Risk Estimator")
    st.write("Answer questions 73–80 to predict your exposure category.")

    # Collect answers for questions 73 to 80
    q1 = st.selectbox("73. How many ounces of bottled water do you drink each day, on average?", [
        "Over 50.8 ounces", "34-50.7 ounces", "17-33.9 ounces", "1-16.9 ounces", "Less than 1 ounce"])
    q2 = st.selectbox("74. How many ounces of water do you typically drink each day from refillable plastic water bottles?", [
        "Over 96 ounces", "65-96 ounces", "33-64 ounces", "1-32 ounces", "Less than 1 ounce"])
    q3 = st.selectbox("75. How many ounces of other beverages (juice, soda, sports drinks, etc.) do you typically drink each day from plastic packaging?", [
        "Over 40 ounces", "20-40 ounces", "1-19 ounces", "Less than 1 ounce"])
    q4 = st.selectbox("76. How often, on average each day, do you use disposable cups for hot drinks (e.g., tea, coffee)?", [
        "5 or more times per day", "3-4 times per day", "1-2 times per day", "Less than 1 time per day", "Never"])
    q5 = st.selectbox("77. How often, on average each day, do you use a microwave to heat water or beverages in a disposable cup?", [
        "5 or more times per day", "3-4 times per day", "1-2 times per day", "Less than 1 time per day", "Never"])
    q6 = st.selectbox("78. During the summer, how often each week do you leave bottled water in a car or under direct sunlight for over 30 minutes?", [
        "5 or more days per week", "3-4 days per week", "1-2 days per week", "Never"])
    q7 = st.selectbox("79. During the summer, how often each week do you leave refillable plastic water bottles in a car or under direct sunlight?", [
        "5 or more days per week", "3-4 days per week", "1-2 days per week", "Never"])
    q8 = st.selectbox("80. During the summer, how often each week do you leave other beverages with plastic packaging in a car or under direct sunlight?", [
        "5 or more days per week", "3-4 days per week", "1-2 days per week", "Never"])

    # ---- Map Answers to Numerical Values ----
    map_5pt = {
        # Q73
        "Over 50.8 ounces": 5,
        "34-50.7 ounces": 4,
        "17-33.9 ounces": 3,
        "1-16.9 ounces": 2,
        "Less than 1 ounce": 1,
        
        # Q74
        "Over 96 ounces": 5,
        "65-96 ounces": 4,
        "33-64 ounces": 3,
        "1-32 ounces": 2,
        "Less than 1 ounce": 1,

        # Q76/Q77
        "5 or more times per day": 5,
        "3-4 times per day": 4,
        "1-2 times per day": 3,
        "Less than 1 time per day": 2,
        "Never": 1
    }
    map_4pt = {
        "Over 40 ounces": 4,
        "20-40 ounces": 3,
        "1-19 ounces": 2,
        "Less than 1 ounce": 1,
        "5 or more days per week": 4,
        "3-4 days per week": 3,
        "1-2 days per week": 2,
        "Never": 1
    }

    # Convert selections to numbers
    responses = [
        map_5pt[q1],
        map_5pt[q2],
        map_4pt[q3],
        map_5pt[q4],
        map_5pt[q5],
        map_4pt[q6],
        map_4pt[q7],
        map_4pt[q8]
    ]

# ---- Predict on Button Click ----
    if st.button("🔍 Predict Exposure Category"):
        responses = [
            map_5pt[q1], map_5pt[q2], map_4pt[q3], map_5pt[q4],
            map_5pt[q5], map_4pt[q6], map_4pt[q7], map_4pt[q8]
        ]
        # Scale and predict
        X_input_scaled = scaler.transform([responses])
        probs = model.predict(X_input_scaled)[0]
        predicted_class_idx = int(np.argmax(probs))
        predicted_class = class_names[predicted_class_idx]
        st.success(f"🧬 Your predicted microplastic exposure category is: **{predicted_class}**")

        # Sum up user's exposure score
        user_score = sum(responses)
        st.subheader("📊 Your Score Compared to the Population")

        
            # Create histogram trace
        hist_trace = go.Histogram(
            x=df_scores['Total_Exposure_Score'],
            nbinsx=30,
            marker=dict(color='mediumpurple', line=dict(color='black', width=1)),
            opacity=0.7,
            name='Population Scores'
        )

        # Add vertical line for the user's score
        line_trace = go.Scatter(
            x=[user_score, user_score],
            y=[0, df_scores['Total_Exposure_Score'].value_counts().max()],
            mode="lines+text",
            name=f"Your Score: {user_score}",
            line=dict(color="crimson", width=3, dash="dash"),
            text=[f"Your Score: {user_score}"],
            textposition="top right"
        )

        # Combine both traces
        fig = go.Figure(data=[hist_trace, line_trace])

        # Customize layout
        fig.update_layout(
            title="📊 Distribution of Exposure Scores in the Population",
            xaxis_title="Exposure Score",
            yaxis_title="Count",
            template="plotly_white",
            legend=dict(x=0.7, y=0.95),
            hovermode="x"
        )

        # Show in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Optional percentile
        percentile = (df_scores['Exposure_Score'] < user_score).mean() * 100
        st.markdown(f"🔢 Your score of **{user_score}** is higher than **{percentile:.1f}%** of the population surveyed in Nebraska.")

# -------------------------
# 🗺️ PAGE 2: Nebraska Map
# -------------------------
elif page == "🗺️ County Risk Map":
    st.title("🗺️ Simulated Microplastic Exposure Score by Nebraska County")

    import json
    from urllib.request import urlopen

    # Load GeoJSON
    with urlopen("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json") as r:
        counties_geo = json.load(r)

    # Ensure FIPS is string
    df_map["fips"] = df_map["fips"].astype(str)

    # Assign Risk Level
    def assign_risk(score):
        if score >= 16: return "High"
        elif score >= 15: return "High‑Medium"
        elif score >= 13: return "Medium"
        elif score >= 11: return "Low‑Medium"
        else: return "Low"
    df_map["Risk_Level"] = df_map["mean_score"].apply(assign_risk)

    # Filter for Nebraska (FIPS starting with '31')
    df_ne = df_map[df_map["fips"].str.startswith("31")]

    # Plot
    fig = px.choropleth(
        df_ne,
        geojson=counties_geo,
        locations="fips",
        color="mean_score",
        scope="usa",
        color_continuous_scale="Viridis",
        range_color=(8, 36),
        labels={'mean_score': 'Mean Exposure'},
        hover_data={"county": True, "mean_score": False, "Risk_Level": True, "fips": False},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        title="📍 Simulated Microplastic Exposure Score by Nebraska County",
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    st.plotly_chart(fig, use_container_width=True)


