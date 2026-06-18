import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.io import loadmat

# Page Config
st.set_page_config(
    page_title="BatteryIQ Pro",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab"] { color: #00ff88; }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #00ff88;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔋 BatteryIQ Pro — RUL Prediction Dashboard")
st.markdown("---")
# Load Data
@st.cache_data
def load_data():
    battery_files = {
        'B0005': 'data/1. BatteryAgingARC-FY08Q4/B0005.mat',
        'B0006': 'data/1. BatteryAgingARC-FY08Q4/B0006.mat',
        'B0007': 'data/1. BatteryAgingARC-FY08Q4/B0007.mat',
        'B0018': 'data/1. BatteryAgingARC-FY08Q4/B0018.mat'
    }

    all_dfs = []

    for battery_name, file_path in battery_files.items():
        bat = loadmat(file_path)
        b = bat[battery_name][0][0]
        cycle = b['cycle'][0]

        capacities = []
        cycle_numbers = []
        count = 1

        for i in range(len(cycle)):
            if cycle[i]['type'][0] == 'discharge':
                capacity = cycle[i]['data'][0][0]['Capacity'][0][0]
                capacities.append(capacity)
                cycle_numbers.append(count)
                count += 1

        temp_df = pd.DataFrame({
            'Battery': battery_name,
            'Cycle': cycle_numbers,
            'Capacity': capacities
        })
        all_dfs.append(temp_df)

    df = pd.concat(all_dfs, ignore_index=True)
    return df

df = load_data()

# Load Models
@st.cache_resource
def load_models():
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/xgboost_rul_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    return rf_model, xgb_model

rf_model, xgb_model = load_models()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Degradation Analysis",
    "📊 Model Performance",
    "🔮 Live RUL Prediction",
    "🔍 Feature Analysis",
    "🚗 Fleet Comparison"
])
# Tab 1 - Degradation Analysis
with tab1:
    st.header("Battery Capacity Degradation Analysis")

    col1, col2, col3, col4 = st.columns(4)

    for col, battery in zip([col1, col2, col3, col4], ['B0005', 'B0006', 'B0007', 'B0018']):
        bat_df = df[df['Battery'] == battery]
        initial = bat_df['Capacity'].max()
        final = bat_df['Capacity'].min()
        fade = ((initial - final) / initial) * 100
        with col:
            st.metric(f"🔋 {battery}", f"{final:.3f} Ah", f"-{fade:.1f}%")

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    colors = ['#00ff88', '#00aaff', '#ff6600', '#ff0066']

    for battery, color in zip(['B0005', 'B0006', 'B0007', 'B0018'], colors):
        bat_df = df[df['Battery'] == battery]
        ax.plot(bat_df['Cycle'], bat_df['Capacity'], label=battery, color=color, linewidth=2)

    ax.set_xlabel('Cycle Number', color='white')
    ax.set_ylabel('Capacity (Ah)', color='white')
    ax.set_title('Capacity Degradation Curve', color='white')
    ax.tick_params(colors='white')
    ax.legend()
    ax.grid(True, alpha=0.2)

    st.pyplot(fig)
    # Tab 2 - Model Performance
with tab2:
    st.header("Model Performance Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌲 Random Forest MAE", "7.43")
    with col2:
        st.metric("🌲 Random Forest RMSE", "9.44")
    with col3:
        st.metric("🏆 Best Model", "Random Forest")

    st.markdown("---")

    # Bar chart comparison
    models = ['Random Forest', 'XGBoost', 'LSTM']
    mae_scores = [7.43, 7.93, 10.38]
    rmse_scores = [9.44, 9.68, 14.24]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

    colors = ['#00ff88', '#00aaff', '#ff6600']

    ax1.bar(models, mae_scores, color=colors)
    ax1.set_title('MAE Comparison')
    ax1.set_ylabel('MAE')

    ax2.bar(models, rmse_scores, color=colors)
    ax2.set_title('RMSE Comparison')
    ax2.set_ylabel('RMSE')

    st.pyplot(fig)
    # Tab 3 - Live RUL Prediction
with tab3:
    st.header("🔮 Live RUL Prediction")
    st.markdown("Adjust the sliders to predict Remaining Useful Life")

    col1, col2 = st.columns(2)

    with col1:
        cycle = st.slider("Cycle Number", 1, 168, 50)
        capacity = st.slider("Capacity (Ah)", 1.0, 2.0, 1.8, step=0.001)
        capacity_fade = st.slider("Capacity Fade (Ah)", 0.0, 0.5, 0.05, step=0.001)

    with col2:
        soh = st.slider("State of Health (SOH)", 0.5, 1.0, 0.95, step=0.01)
        roll_mean = st.slider("Capacity Roll Mean", 1.0, 2.0, 1.8, step=0.001)
        fade_rate = st.slider("Fade Rate", 0.0, 0.05, 0.01, step=0.001)

    input_data = np.array([[cycle, capacity, capacity_fade, soh, roll_mean, fade_rate]])

    rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🌲 Random Forest RUL", f"{max(0, int(rf_pred))} cycles")
    with col2:
        st.metric("⚡ XGBoost RUL", f"{max(0, int(xgb_pred))} cycles")

    # Health indicator
    if soh >= 0.9:
        st.success("🟢 Battery Health: GOOD")
    elif soh >= 0.8:
        st.warning("🟡 Battery Health: DEGRADING")
    else:
        st.error("🔴 Battery Health: CRITICAL")
        # Tab 4 - Feature Analysis
with tab4:
    st.header("🔍 Feature Analysis")

    # Feature Importance
    features = ['Cycle', 'Capacity', 'Capacity_Fade', 'SOH', 'Capacity_RollMean', 'Fade_Rate']
    importance = rf_model.feature_importances_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    # Feature importance bar chart
    colors = ['#00ff88' if i == importance.argmax() else '#00aaff' for i in range(len(features))]
    ax1.barh(features, importance, color=colors)
    ax1.set_title('Feature Importance (Random Forest)')
    ax1.set_xlabel('Importance Score')

    # Correlation heatmap
    all_processed = []
    for battery_name, group in df.groupby('Battery'):
        group = group.copy()
        initial_capacity = group['Capacity'].max()
        eol_threshold = 0.80 * initial_capacity
        eol_cycle = group[group['Capacity'] <= eol_threshold]['Cycle'].min()
        group['Capacity_Fade'] = initial_capacity - group['Capacity']
        group['SOH'] = group['Capacity'] / initial_capacity
        group['RUL'] = eol_cycle - group['Cycle']
        group = group[group['RUL'] >= 0]
        all_processed.append(group)

    df_processed = pd.concat(all_processed, ignore_index=True)

    corr = df_processed[['Cycle', 'Capacity', 'Capacity_Fade', 'SOH', 'RUL']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title('Feature Correlation Matrix')

    st.pyplot(fig)
    # Tab 5 - Fleet Comparison
with tab5:
    st.header("🚗 Fleet Battery Comparison")

    fleet_data = []
    for battery_name, group in df.groupby('Battery'):
        initial = group['Capacity'].max()
        final = group['Capacity'].min()
        eol_threshold = 0.80 * initial
        eol_cycle = group[group['Capacity'] <= eol_threshold]['Cycle'].min()
        fade_pct = ((initial - final) / initial) * 100
        current_soh = final / initial

        fleet_data.append({
            'Battery': battery_name,
            'Initial Capacity': round(initial, 4),
            'Current Capacity': round(final, 4),
            'Capacity Fade %': round(fade_pct, 2),
            'Current SOH': round(current_soh, 4),
            'EOL Cycle': eol_cycle if pd.notna(eol_cycle) else 'Not reached'
        })

    fleet_df = pd.DataFrame(fleet_data)
    st.dataframe(fleet_df, use_container_width=True)

    st.markdown("---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')

    colors = ['#00ff88', '#00aaff', '#ff6600', '#ff0066']

    for ax in axes:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.yaxis.label.set_color('white')

    axes[0].bar(fleet_df['Battery'], fleet_df['Capacity Fade %'], color=colors)
    axes[0].set_title('Capacity Fade % per Battery', color='white')
    axes[0].set_ylabel('Fade %')

    axes[1].bar(fleet_df['Battery'], fleet_df['Current SOH'], color=colors)
    axes[1].set_title('Current SOH per Battery', color='white')
    axes[1].set_ylabel('SOH')
    axes[1].set_ylim(0, 1)

    st.pyplot(fig)