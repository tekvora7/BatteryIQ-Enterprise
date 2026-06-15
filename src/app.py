import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="BatteryIQ Enterprise",
    page_icon="🔋",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

with open("models/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("🔋 BatteryIQ Enterprise")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "About Model"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🔋 AI Battery Prognostics Platform")

    st.markdown(
        """
        Predict Remaining Useful Life (RUL) of Lithium-Ion Batteries
        using Machine Learning.
        """
    )

    st.divider()

    left, right = st.columns([1, 1])

    with left:

        st.subheader("Battery Inputs")

        capacity = st.slider(
            "Capacity (Ah)",
            min_value=1.20,
            max_value=1.90,
            value=1.60,
            step=0.01
        )

        capacity_fade = st.slider(
            "Capacity Fade",
            min_value=0.00,
            max_value=0.60,
            value=0.20,
            step=0.01
        )

        soh = st.slider(
            "State of Health",
            min_value=0.60,
            max_value=1.00,
            value=0.85,
            step=0.01
        )

    with right:

        st.subheader("Battery Health")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=soh * 100,
            title={"text": "SOH (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00E676"},
                "steps": [
                    {"range": [0, 60], "color": "#FF5252"},
                    {"range": [60, 80], "color": "#FFC107"},
                    {"range": [80, 100], "color": "#00E676"}
                ]
            }
        ))

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    st.divider()

    if st.button(
        "🚀 Predict Remaining Useful Life",
        use_container_width=True
    ):

        X_live = pd.DataFrame({
            "Capacity": [capacity],
            "Capacity_Fade": [capacity_fade],
            "SOH": [soh]
        })

        prediction = float(
            model.predict(X_live)[0]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted RUL",
            f"{prediction:.0f} Cycles"
        )

        col2.metric(
            "Battery Health",
            f"{soh*100:.1f}%"
        )

        col3.metric(
            "Capacity",
            f"{capacity:.2f} Ah"
        )

        st.divider()

        if soh >= 0.90:
            status = "🟢 Excellent"
            recommendation = "Battery operating optimally."
        elif soh >= 0.75:
            status = "🟡 Good"
            recommendation = "Continue normal monitoring."
        elif soh >= 0.60:
            status = "🟠 Moderate"
            recommendation = "Schedule preventive maintenance."
        else:
            status = "🔴 Critical"
            recommendation = "Immediate replacement recommended."

        st.success(f"Battery Status: {status}")

        st.info(
            f"Recommendation: {recommendation}"
        )

        st.progress(
            int(soh * 100)
        )

        report = pd.DataFrame({
            "Metric": [
                "Capacity",
                "Capacity Fade",
                "SOH",
                "Predicted RUL"
            ],
            "Value": [
                capacity,
                capacity_fade,
                soh,
                prediction
            ]
        })

        st.download_button(
            "📥 Download Report",
            report.to_csv(index=False),
            file_name="battery_report.csv",
            mime="text/csv"
        )

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About Model":

    st.title("📊 Model Information")

    st.subheader("Dataset")

    st.write(
        """
        NASA Lithium-Ion Battery Aging Dataset
        (B0005, B0006, B0007, B0018).
        """
    )

    st.subheader("Features Used")

    feature_df = pd.DataFrame({
        "Feature": [
            "Capacity",
            "Capacity Fade",
            "State of Health (SOH)"
        ]
    })

    st.table(feature_df)

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        "4.10"
    )

    col2.metric(
        "RMSE",
        "6.83"
    )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": [
            "Capacity",
            "Capacity Fade",
            "SOH"
        ],
        "Importance": [
            0.352,
            0.349,
            0.298
        ]
    })

    st.bar_chart(
        importance.set_index(
            "Feature"
        )
    )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "BatteryIQ Enterprise | AI-Based Battery Remaining Useful Life Prediction"
)