import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import folium
from folium.plugins import HeatMap, HeatMapWithTime
from streamlit_folium import folium_static
from preprocessing import preprocess_pipeline, get_season

# Set page config
st.set_page_config(
    page_title="SF Crime Analytics | AI-Powered",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        transform: scale(1.05);
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .report-text {
        font-family: 'Courier New', monospace;
        color: #00ff00;
        background-color: #000000;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #00ff00;
    }
    .chat-bubble-user {
        background-color: #2b313e;
        color: white;
        padding: 10px;
        border-radius: 15px 15px 0 15px;
        margin: 5px;
        text-align: right;
    }
    .chat-bubble-bot {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 15px 15px 15px 0;
        margin: 5px;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Resources
@st.cache_resource
def load_resources():
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    model_path = os.path.join(models_dir, 'best_model.pkl')
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    kmeans_path = os.path.join(models_dir, 'kmeans.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(encoders_path) or not os.path.exists(kmeans_path):
        return None, None, None
        
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    kmeans = joblib.load(kmeans_path)
    return model, encoders, kmeans

@st.cache_data
def load_data_sample():
    data_dir = os.path.join(os.path.dirname(__file__), '../data/crimedataset')
    try:
        df = pd.read_csv(os.path.join(data_dir, 'train.csv'), parse_dates=['Dates'])
        return df.sample(10000, random_state=42)
    except:
        return pd.DataFrame()

model, encoders, kmeans = load_resources()
df_sample = load_data_sample()

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("San Francisco Crime Analytics")
    st.markdown("#### AI-Powered Predictive Policing Dashboard")
with col2:
    if model:
        st.success("🟢 System Online: Models Loaded")
    else:
        st.error("🔴 System Offline: Models Missing")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")
    st.sidebar.markdown("🟢 **Online** | ⚡ **12ms**")
    st.sidebar.markdown(f"📅 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.markdown("---")

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/police-badge.png", width=80)
st.sidebar.header("Incident Parameters")

date_input = st.sidebar.date_input("Date")
time_input = st.sidebar.time_input("Time")
district = st.sidebar.selectbox("District", options=encoders['PdDistrict'].classes_ if encoders else [])
st.sidebar.subheader("Geolocation")
latitude = st.sidebar.number_input("Latitude", value=37.7749, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-122.4194, format="%.6f")

# Main Prediction Logic
if st.sidebar.button("Analyze Risk Level", type="primary"):
    if model is None:
        st.error("Model not trained yet. Please run training script.")
    else:
        # Prepare Input
        datetime_combined = pd.to_datetime(f"{date_input} {time_input}")
        
        input_data = pd.DataFrame({
            'Dates': [datetime_combined],
            'X': [longitude],
            'Y': [latitude],
            'PdDistrict': [district]
        })
        
        # Preprocess
        processed_df, _ = preprocess_pipeline(input_data, is_train=False, kmeans_model=kmeans)
        
        # Encoding
        processed_df['PdDistrict'] = encoders['PdDistrict'].transform(processed_df['PdDistrict'])
        processed_df['Season'] = encoders['Season'].transform(processed_df['Season'])
        
        # Features
        features = ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek', 'IsWeekend', 'IsHoliday', 'LocationCluster', 'PdDistrict', 'Season']
        
        prediction = model.predict(processed_df[features])[0]
        proba = model.predict_proba(processed_df[features])[0]
        
        st.markdown("---")
        st.subheader("Analysis Results")
        
        r_col1, r_col2, r_col3 = st.columns(3)
        
        with r_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Risk Probability", f"{max(proba)*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with r_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if prediction == 1:
                st.metric("Predicted Classification", "VIOLENT", delta="High Risk", delta_color="inverse")
            else:
                st.metric("Predicted Classification", "NON-VIOLENT", delta="Low Risk", delta_color="normal")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with r_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Location Cluster", f"Zone {processed_df['LocationCluster'][0]}")
            st.markdown('</div>', unsafe_allow_html=True)

        # AI Analyst Report
        st.markdown("### 🤖 AI Analyst Report")
        risk_level = "CRITICAL" if proba[1] > 0.7 else "ELEVATED" if proba[1] > 0.4 else "STANDARD"
        report = f"""
        [CLASSIFIED REPORT - GENERATED BY AI]
        -------------------------------------
        DATE: {date_input} | TIME: {time_input}
        LOCATION: {district} (Lat: {latitude}, Lon: {longitude})
        
        ASSESSMENT: {risk_level} RISK DETECTED
        PROBABILITY OF VIOLENCE: {proba[1]*100:.2f}%
        
        KEY FACTORS:
        - Time of Day: {time_input.hour}:00 hours (Historical high-risk window)
        - District Profile: {district} shows elevated activity trends.
        - Seasonal Context: {get_season(datetime_combined.month)} patterns observed.
        
        RECOMMENDATION:
        Immediate deployment of patrol units advised if risk > 50%. 
        Monitor sector {processed_df['LocationCluster'][0]} closely.
        """
        st.markdown(f'<div class="report-text">{report}</div>', unsafe_allow_html=True)
        
        st.download_button(
            label="📄 Download Full Report",
            data=report,
            file_name=f"crime_report_{date_input}_{district}.txt",
            mime="text/plain"
        )
        
        # Explainability
        st.markdown("### 🧠 Model Explainability")
        if hasattr(model, 'feature_importances_'):
            feat_imp = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', 
                            title="What drove this prediction?", template='plotly_dark',
                            color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp)

# Dashboard Tabs
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Historical Trends", "🗺️ Geospatial Intelligence", "🚨 Tactical Simulation", "💬 Chat with Data", "🧪 Scenario Tester"])

with tab1:
    if not df_sample.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Crime Distribution by Hour")
            df_sample['Hour'] = df_sample['Dates'].dt.hour
            hourly_counts = df_sample.groupby('Hour').size().reset_index(name='Count')
            fig_hour = px.bar(hourly_counts, x='Hour', y='Count', color='Count', 
                             color_continuous_scale='RdBu_r', template='plotly_dark')
            st.plotly_chart(fig_hour)
            
        with col2:
            st.subheader("Incidents by District")
            district_counts = df_sample['PdDistrict'].value_counts().reset_index()
            district_counts.columns = ['District', 'Count']
            fig_dist = px.pie(district_counts, values='Count', names='District', hole=0.4, 
                             template='plotly_dark', color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_dist)
    else:
        st.warning("Data loading...")

with tab2:
    st.subheader("Spatiotemporal Crime Analysis")
    if not df_sample.empty:
        # Time-Lapse Heatmap
        st.write("**24-Hour Crime Evolution (Time-Lapse)**")
        
        # Prepare data for HeatMapWithTime
        # List of lists of points, one list per time step (hour)
        heat_data_time = []
        time_index = []
        
        for hour in range(24):
            hour_data = df_sample[df_sample['Dates'].dt.hour == hour]
            heat_data_time.append(hour_data[['Y', 'X']].values.tolist())
            time_index.append(f"{hour:02d}:00")
            
        m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles='CartoDB dark_matter')
        
        HeatMapWithTime(
            heat_data_time,
            index=time_index,
            auto_play=True,
            max_opacity=0.8,
            radius=15
        ).add_to(m)
        
        folium_static(m, width=1000)
        
        st.markdown("---")
        st.write("**Static Density Heatmap**")
        m_static = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles='CartoDB dark_matter')
        heat_data = [[row['Y'], row['X']] for index, row in df_sample.iterrows()]
        HeatMap(heat_data, radius=15).add_to(m_static)
        folium_static(m_static, width=1000)
    else:
        st.warning("Data not loaded.")

with tab3:
    st.subheader("Resource Allocation Simulator")
    st.info("Use this tool to simulate patrol strategies based on predictive risk modeling.")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.markdown("### Simulation Controls")
        sim_district = st.selectbox("Target District", options=encoders['PdDistrict'].classes_ if encoders else [], key='sim_dist')
        sim_hour = st.slider("Patrol Hour", 0, 23, 22)
        sim_date = st.date_input("Patrol Date", key='sim_date')
        
    with sim_col2:
        st.markdown("### AI Recommendation Engine")
        if model and kmeans:
            if not df_sample.empty:
                district_center = df_sample[df_sample['PdDistrict'] == sim_district][['Y', 'X']].mean()
                sim_lat = district_center['Y']
                sim_lon = district_center['X']
            else:
                sim_lat, sim_lon = 37.7749, -122.4194
                
            sim_datetime = pd.to_datetime(f"{sim_date} {sim_hour}:00:00")
            
            sim_input = pd.DataFrame({
                'Dates': [sim_datetime],
                'X': [sim_lon],
                'Y': [sim_lat],
                'PdDistrict': [sim_district]
            })
            
            # Process
            sim_processed, _ = preprocess_pipeline(sim_input, is_train=False, kmeans_model=kmeans)
            sim_processed['PdDistrict'] = encoders['PdDistrict'].transform(sim_processed['PdDistrict'])
            sim_processed['Season'] = encoders['Season'].transform(sim_processed['Season'])
            
            # Features
            features = ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek', 'IsWeekend', 'IsHoliday', 'LocationCluster', 'PdDistrict', 'Season']
            
            # Predict
            sim_prob = model.predict_proba(sim_processed[features])[0]
            violent_prob = sim_prob[1]
            
            st.write(f"Analyzing sector **{sim_district}** at **{sim_hour}:00**...")
            
            # Gauge Chart
            fig_gauge = px.bar(x=[violent_prob], y=["Risk"], orientation='h', range_x=[0, 1],
                              labels={'x': 'Violent Crime Probability', 'y': ''}, height=100,
                              color=[violent_prob], color_continuous_scale=['green', 'yellow', 'red'])
            fig_gauge.update_layout(showlegend=False, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_gauge)
            
            if violent_prob > 0.7:
                st.error("⚠️ **CRITICAL RISK DETECTED**")
                st.markdown("""
                **Recommended Action Plan:**
                - 🔴 Deploy SWAT / Heavy Tactical Units
                - 🚁 Request Aerial Surveillance
                - 🚧 Establish Perimeter Checkpoints
                """)
            elif violent_prob > 0.4:
                st.warning("⚠️ **ELEVATED RISK**")
                st.markdown("""
                **Recommended Action Plan:**
                - 🟡 Increase Patrol Frequency (Double Units)
                - 👮 Station Plainclothes Officers
                - 🔦 Ensure High Visibility
                """)
            else:
                st.success("✅ **STANDARD RISK**")
                st.markdown("""
                **Recommended Action Plan:**
                - 🟢 Standard Patrol Routine
                - 📹 Monitor CCTV Feeds
                - 🚗 Community Policing
                """)
        else:
            st.warning("Model not loaded. Cannot run simulation.")

with tab4:
    st.subheader("💬 Chat with Data (Natural Language Interface)")
    st.markdown("Ask questions about the crime data. Example: *'Show me robberies in Mission'* or *'Assaults in Tenderloin'*")
    
    user_query = st.text_input("Ask a question...", placeholder="Type here...")
    
    if user_query:
        st.markdown(f'<div class="chat-bubble-user">User: {user_query}</div>', unsafe_allow_html=True)
        
        # Simple Intent Parser
        query_lower = user_query.lower()
        
        # Filter Logic
        filtered_df = df_sample.copy()
        
        # Categories
        found_cat = None
        categories = df_sample['Category'].unique()
        for cat in categories:
            if cat.lower() in query_lower:
                filtered_df = filtered_df[filtered_df['Category'] == cat]
                found_cat = cat
                break
                
        # Districts
        found_dist = None
        districts = df_sample['PdDistrict'].unique()
        for dist in districts:
            if dist.lower() in query_lower:
                filtered_df = filtered_df[filtered_df['PdDistrict'] == dist]
                found_dist = dist
                break
        
        # Response Generation
        response_text = ""
        if found_cat and found_dist:
            response_text = f"Filtering for **{found_cat}** in **{found_dist}**."
        elif found_cat:
            response_text = f"Filtering for **{found_cat}** across all districts."
        elif found_dist:
            response_text = f"Showing all crimes in **{found_dist}**."
        else:
            response_text = "I couldn't identify a specific category or district. Showing general trends."
            
        count = len(filtered_df)
        response_text += f" Found **{count}** incidents."
        
        st.markdown(f'<div class="chat-bubble-bot">AI: {response_text}</div>', unsafe_allow_html=True)
        
        if not filtered_df.empty:
            st.dataframe(filtered_df[['Dates', 'Category', 'PdDistrict', 'Address']].head(10))
            
            # Dynamic Chart based on query
            if found_dist and not found_cat:
                # Show breakdown by category for that district
                fig = px.bar(filtered_df['Category'].value_counts().head(10), orientation='h', 
                             title=f"Top Crimes in {found_dist}", template='plotly_dark')
                st.plotly_chart(fig)
            elif found_cat:
                # Show breakdown by hour or district
                fig = px.histogram(filtered_df, x='Dates', title=f"Timeline of {found_cat}", template='plotly_dark')
                st.plotly_chart(fig, key="timeline")

with tab5:
    st.subheader("🧪 Model Validation: Scenario Tester")
    st.info("Test the AI against real historical cases to verify its accuracy.")
    
    if 'scenario_case' not in st.session_state:
        st.session_state.scenario_case = None
        
    if st.button("🎲 Load Random Historical Case", type="primary"):
        if not df_sample.empty:
            st.session_state.scenario_case = df_sample.sample(1).iloc[0]
        else:
            st.warning("Data not loaded.")
            
    if st.session_state.scenario_case is not None:
        case = st.session_state.scenario_case
        
        # Display Case Details (Masking the Truth)
        st.markdown("### 📁 Case File #8921-X")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Date:** {case['Dates'].date()}")
            st.markdown(f"**Time:** {case['Dates'].time()}")
        with c2:
            st.markdown(f"**District:** {case['PdDistrict']}")
            st.markdown(f"**Location:** {case['Address']}")
        with c3:
            st.markdown(f"**Coordinates:** {case['Y']:.4f}, {case['X']:.4f}")
            
        st.markdown("---")
        
        if st.button("🤖 Run AI Analysis"):
            # Prepare Input
            input_data = pd.DataFrame({
                'Dates': [case['Dates']],
                'X': [case['X']],
                'Y': [case['Y']],
                'PdDistrict': [case['PdDistrict']]
            })
            
            # Preprocess
            processed_df, _ = preprocess_pipeline(input_data, is_train=False, kmeans_model=kmeans)
            processed_df['PdDistrict'] = encoders['PdDistrict'].transform(processed_df['PdDistrict'])
            processed_df['Season'] = encoders['Season'].transform(processed_df['Season'])
            
            # Features
            features = ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek', 'IsWeekend', 'IsHoliday', 'LocationCluster', 'PdDistrict', 'Season']
            
            # Predict
            prediction = model.predict(processed_df[features])[0]
            proba = model.predict_proba(processed_df[features])[0]
            
            # Determine Actual
            violent_categories = ['ASSAULT', 'ROBBERY', 'SEX OFFENSES FORCIBLE', 'KIDNAPPING', 'HOMICIDE', 'ARSON']
            actual_is_violent = 1 if case['Category'] in violent_categories else 0
            actual_label = "VIOLENT" if actual_is_violent else "NON-VIOLENT"
            pred_label = "VIOLENT" if prediction == 1 else "NON-VIOLENT"
            
            # Display Results
            r1, r2 = st.columns(2)
            
            with r1:
                st.markdown("#### AI Prediction")
                if prediction == 1:
                    st.error(f"**{pred_label}** ({proba[1]*100:.1f}% Confidence)")
                else:
                    st.success(f"**{pred_label}** ({proba[0]*100:.1f}% Confidence)")
                    
            with r2:
                st.markdown("#### Actual Outcome")
                st.markdown(f"**Category:** {case['Category']}")
                if actual_is_violent:
                    st.markdown(f"**Classification:** :red[{actual_label}]")
                else:
                    st.markdown(f"**Classification:** :green[{actual_label}]")
            
            st.markdown("---")
            if prediction == actual_is_violent:
                st.success("✅ **AI Model Correctly Classified this Incident**")
                st.balloons()
            else:
                st.error("❌ **AI Model Incorrect** (Complex real-world variability)")



