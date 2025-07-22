import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date

# --- PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="Underwriting Co-Pilot",
    page_icon="🚀"
)

# --- MOCK DATA GENERATION ---
@st.cache_data
def generate_mock_data():
    """Generates a mock dataset for the insurance portfolio."""
    np.random.seed(42)
    num_users = 100
    regions = ["North-East", "West", "South", "Midwest"]
    vehicle_types = ["Sedan", "SUV", "Truck", "Sports Car"]
    
    data = {
        "user_id": [f"C-{10234 + i}" for i in range(num_users)],
        "region": np.random.choice(regions, num_users),
        "vehicle_type": np.random.choice(vehicle_types, num_users),
        "loyalty_tier": np.random.choice(["Gold", "Silver", "Bronze"], num_users, p=[0.2, 0.5, 0.3]),
        "age": np.random.randint(20, 70, num_users),
        "years_driving": np.random.randint(2, 50, num_users),
        "recent_claims": np.random.randint(0, 4, num_users),
        "vehicle_value": np.random.randint(15000, 80000, num_users),
    }
    df = pd.DataFrame(data)

    # Generate historical data for each user
    history = []
    for user_id in df["user_id"]:
        for year in range(2022, 2025):
            history.append({
                "user_id": user_id,
                "year": year,
                "premium": np.random.randint(800, 5000),
                "claims_cost": np.random.choice([0, np.random.randint(200, 10000)], p=[0.7, 0.3])
            })
    history_df = pd.DataFrame(history)
    
    return df, history_df

# --- SIMULATED MODEL ---
def simulate_prediction(user_data):
    """Simulates ML model prediction and SHAP explanation."""
    base_premium = 1000
    
    # Simple logic for prediction
    premium = (base_premium + 
               (user_data["vehicle_value"] * 0.01) - 
               (user_data["years_driving"] * 10) + 
               (user_data["recent_claims"] * 500))
    
    claim_cost = user_data["recent_claims"] * np.random.uniform(400, 600)
    
    # Simulate SHAP values
    shap_values = {
        "Recent Claims": user_data["recent_claims"] * 500,
        "Vehicle Value": user_data["vehicle_value"] * 0.01,
        "Years Driving": -user_data["years_driving"] * 10,
        "Region (West)": -50 if user_data["region"] == "West" else 20
    }
    
    # Simulate fraud/retention flags
    fraud_risk = "Low" if user_data["recent_claims"] < 2 else "High"
    retention_risk = "Medium" if user_data["loyalty_tier"] == "Bronze" else "Low"
    
    return {
        "premium": int(premium),
        "claim_cost": int(claim_cost),
        "discount_suggestion": 0.08 if retention_risk == "Low" else 0.03,
        "shap_values": shap_values,
        "fraud_risk": fraud_risk,
        "retention_risk": retention_risk
    }

# --- PLOTTING FUNCTIONS ---
def plot_shap_values(shap_values):
    """Creates a horizontal bar chart to simulate a SHAP force plot."""
    shap_df = pd.DataFrame(list(shap_values.items()), columns=["Feature", "Impact"])
    shap_df["Positive"] = shap_df["Impact"] > 0
    
    chart = alt.Chart(shap_df).mark_bar().encode(
        x=alt.X('Impact:Q', title="Impact on Premium ($)"),
        y=alt.Y('Feature:N', sort='-x'),
        color=alt.Color('Positive:N', scale=alt.Scale(domain=[True, False], range=['#d62728', '#2ca02c']), legend=None)
    ).properties(
        title="Feature Contribution to Premium"
    )
    return chart

# --- MAIN APP ---

# Load data
df, history_df = generate_mock_data()

# --- HEADER ---
st.title("🚀 Underwriting Co-Pilot")
st.markdown("A unified dashboard for client analysis, portfolio monitoring, and model governance.")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🔍 Client Analyzer", "📊 Portfolio Summary", "⚙️ Model & Tuning Console"])

# --- TAB 1: CLIENT ANALYZER ---
with tab1:
    st.header("Analyze a Client")
    
    # --- INPUTS ---
    col1, col2 = st.columns([1, 3])
    with col1:
        user_id_list = df["user_id"].unique()
        selected_user_id = st.selectbox("Select User ID", options=user_id_list, index=0)
        analysis_date = st.date_input("Date of Analysis", value=date.today())
    
    # --- ANALYSIS BUTTON ---
    if st.button("⚡ Analyze Client", type="primary", use_container_width=True):
        st.success(f"Successfully analyzed client **{selected_user_id}** for date **{analysis_date}**.")
        
        user_data = df[df["user_id"] == selected_user_id].iloc[0]
        user_history = history_df[history_df["user_id"] == selected_user_id]
        prediction = simulate_prediction(user_data)
        
        # --- KEY OUTPUTS (METRICS) ---
        st.subheader("Key Predictive Insights")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        last_year_premium = user_history[user_history["year"] == 2024]["premium"].iloc[0]
        premium_delta = prediction["premium"] - last_year_premium
        
        last_year_claims = user_history[user_history["year"] == 2024]["claims_cost"].iloc[0]
        claim_delta = prediction["claim_cost"] - last_year_claims
        
        metric_col1.metric(label="Predicted Annual Premium", value=f"${prediction['premium']:,}", delta=f"${premium_delta:,} vs. last year")
        metric_col2.metric(label="Predicted Claim Cost", value=f"${prediction['claim_cost']:,}", delta=f"${claim_delta:,} vs. last year", delta_color="inverse")
        metric_col3.metric(label="Suggested Discount", value=f"{prediction['discount_suggestion']:.0%}", delta="Based on Retention Score")

        st.divider()

        # --- PROFILE & FLAGS ---
        profile_col, flags_col = st.columns([2, 1.5])
        with profile_col:
            st.subheader("Client Profile")
            st.text(f"""
Region: {user_data['region']}
Vehicle: {user_data['vehicle_type']}
Loyalty Tier: {user_data['loyalty_tier']} ({user_data['years_driving']} years)
            """)
        with flags_col:
            st.subheader("Risk Flags")
            if prediction["retention_risk"] == "Medium":
                st.warning("⚠️ **Retention Risk:** Medium probability of churn at renewal.")
            else:
                st.success("✅ **Retention Risk:** Low churn probability.")
            
            if prediction["fraud_risk"] == "High":
                st.error("🚩 **Fraud Alert:** High probability of fraud detected.")
            else:
                st.success("✅ **Fraud Risk:** Low fraud probability.")

        # --- DETAILED INSIGHTS (SUB-TABS) ---
        detail_tabs = st.tabs(["📈 Trends", "🧠 Explainability (XAI)"])
        
        with detail_tabs[0]:
            st.subheader("Historical Trends")
            
            # Premium Trend Chart
            premium_chart = alt.Chart(user_history).mark_line(point=True).encode(
                x=alt.X('year:O', title='Year'),
                y=alt.Y('premium:Q', title='Annual Premium ($)'),
                tooltip=['year', 'premium']
            ).properties(title="Premium History")
            st.altair_chart(premium_chart, use_container_width=True)

            # Claims Trend Chart
            claims_chart = alt.Chart(user_history).mark_bar().encode(
                x=alt.X('year:O', title='Year'),
                y=alt.Y('claims_cost:Q', title='Total Claims Cost ($)'),
                tooltip=['year', 'claims_cost']
            ).properties(title="Claims History")
            st.altair_chart(claims_chart, use_container_width=True)

        with detail_tabs[1]:
            st.subheader("What's driving the premium?")
            st.write("The plot below shows which factors contributed most to the prediction.")
            shap_chart = plot_shap_values(prediction["shap_values"])
            st.altair_chart(shap_chart, use_container_width=True)
            
            with st.expander("View Textual Explanation"):
                st.info("""
                **Textual Explanation:**
                - **Factors Increasing Premium:** Positive values (red bars) push the premium higher than the baseline. Common factors include recent claims or high-value vehicles.
                - **Factors Decreasing Premium:** Negative values (green bars) lower the premium. This is often due to a long, clean driving history or residing in a lower-risk region.
                """)

# --- TAB 2: PORTFOLIO SUMMARY ---
with tab2:
    st.header("Portfolio Health at a Glance")
    
    # --- FILTERS in the Sidebar ---
    with st.sidebar:
        st.header("📊 Portfolio Filters")
        selected_region = st.selectbox("Filter by Area", ["All"] + list(df['region'].unique()))
        selected_vehicle = st.multiselect("Filter by Vehicle Type", df['vehicle_type'].unique(), default=df['vehicle_type'].unique())
        
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df['region'] == selected_region]
    filtered_df = filtered_df[filtered_df['vehicle_type'].isin(selected_vehicle)]
    
    # --- GLOBAL KPIs ---
    st.subheader("Key Portfolio Metrics")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    avg_premium = history_df[history_df['user_id'].isin(filtered_df['user_id']) & (history_df['year'] == 2024)]['premium'].mean()
    avg_claims = history_df[history_df['user_id'].isin(filtered_df['user_id']) & (history_df['year'] == 2024)]['claims_cost'].mean()
    loss_ratio = avg_claims / avg_premium if avg_premium > 0 else 0
    
    kpi_col1.metric(label="Average Premium", value=f"${avg_premium:,.2f}")
    kpi_col2.metric(label="Average Claim Cost", value=f"${avg_claims:,.2f}")
    kpi_col3.metric(label="Portfolio Loss Ratio", value=f"{loss_ratio:.2%}")

    st.divider()

    # --- VISUALIZATIONS ---
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Average Premium by Region")
        region_premium = history_df[history_df['year'] == 2024].groupby(df['region'])['premium'].mean().reset_index()
        region_chart = alt.Chart(region_premium).mark_bar().encode(
            x='region:N',
            y='premium:Q'
        ).properties(height=300)
        st.altair_chart(region_chart, use_container_width=True)

    with chart_col2:
        st.subheader("Customer Distribution by Loyalty")
        loyalty_dist = filtered_df['loyalty_tier'].value_counts().reset_index()
        loyalty_chart = alt.Chart(loyalty_dist).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="count", type="quantitative"),
            color=alt.Color(field="loyalty_tier", type="nominal")
        ).properties(height=300)
        st.altair_chart(loyalty_chart, use_container_width=True)

    with st.expander("View Global Feature Importance"):
        st.write("Simulated SHAP summary for the entire portfolio.")
        st.image("https://i.imgur.com/C531xay.png", caption="Mockup of Global SHAP Summary Plot. This shows the general impact of features across all predictions.")

# --- TAB 3: MODEL & TUNING CONSOLE ---
with tab3:
    st.header("Model Governance & Performance")
    col1, col2 = st.columns(2)

    with col1:
        # --- MODEL VERSION INFO ---
        st.subheader("Current Model Details")
        st.info("""
        - **Model Name:** `PremiumPredictor_v2.1-XGBoost`
        - **Version:** `a8c4d2ef-20250715`
        - **Last Trained:** 2025-07-01
        - **Training Data:** `s3://insurance-data/training/2025-q2.parquet`
        """)
        
        # --- EVALUATION METRICS ---
        st.subheader("Evaluation Metrics")
        st.table({
            "Metric": ["RMSE", "R²", "MAE"],
            "Value": ["$124.50", "0.89", "$98.75"]
        })
        
        # --- ACTIONS ---
        if st.button("⚙️ Trigger Model Retraining", help="This would trigger a CI/CD pipeline like Jenkins or GitHub Actions."):
            st.toast("Retraining job sent to ML pipeline...", icon="✅")

    with col2:
        # --- SHAP SUMMARY ---
        st.subheader("Global SHAP Summary Plot")
        st.image("https://i.imgur.com/C531xay.png", caption="Top features influencing predictions across the entire portfolio.")

        # --- LOGS & DATA ---
        st.subheader("Download Artifacts")
        
        # Create dummy data for download
        dummy_logs = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📄 Download Prediction Logs (CSV)",
            data=dummy_logs,
            file_name="prediction_logs.csv",
            mime="text/csv"
        )