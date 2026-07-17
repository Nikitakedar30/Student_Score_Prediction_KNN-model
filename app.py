import streamlit as st
import pickle
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Mentor Map | Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS for a Clean, Modern Look
st.markdown("""
    <style>
    /* Main container background tweaks */
    .main {
        background-color: #f9fbfd;
    }
    /* Title alignment and color */
    .main-title {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #4B5563;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    /* Section headers */
    .section-header {
        color: #2563EB;
        font-weight: 600;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    /* Custom Card for Results */
    .result-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #10B981;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Robust Model Loader
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "model.pkl")
    with open(model_path, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model.pkl: {e}")
    st.info("Make sure 'model.pkl' is placed in the exact same folder as this script.")
    st.stop()

# 4. Header Section
st.markdown("<h1 class='main-title'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze metrics to evaluate and predict student learning outcomes.</p>", unsafe_allow_html=True)

# 5. Interactive Form UI
with st.container(border=True):
    st.markdown("<h3 class='section-header'>📊 Engagement Metrics</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        hours_studied = st.slider(
            "📚 Hours Studied (per week)", 
            min_value=0.0, max_value=100.0, value=10.0, step=0.5,
            help="Total hours the student dedicated to self-study per week."
        )
    with col2:
        attendance_percent = st.slider(
            "🏫 Attendance Rate (%)", 
            min_value=0.0, max_value=100.0, value=85.0, step=1.0,
            help="Overall class attendance percentage."
        )

    st.markdown("<h3 class='section-header'>🧠 Personal & Historical Academic Factors</h3>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        sleep_hours = st.slider(
            "🛌 Average Sleep (per night)", 
            min_value=0.0, max_value=14.0, value=7.0, step=0.5,
            help="Average hours of sleep the student gets per night."
        )
    with col4:
        previous_scores = st.number_input(
            "📝 Previous Exam Score (0-100)", 
            min_value=0.0, max_value=100.0, value=75.0, step=1.0,
            help="The student's score on their last examination."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Styled centered action button
    left_spacer, center_button, right_spacer = st.columns([1, 2, 1])
    with center_button:
        submit_btn = st.button("✨ Run Predictive Analysis", type="primary", use_container_width=True)

# 6. Prediction Logic & Output Styling
if submit_btn:
    input_df = pd.DataFrame([{
        'hours_studied': hours_studied,
        'sleep_hours': sleep_hours,
        'attendance_percent': attendance_percent,
        'previous_scores': previous_scores
    }])
    
    try:
        prediction = model.predict(input_df)[0]
        
        # Displaying an attractive result card
        st.markdown(f"""
            <div class='result-card'>
                <h3 style='margin-top: 0; color: #065F46;'>🎯 Analysis Summary</h3>
                <p style='color: #374151; font-size: 1.05rem; margin-bottom: 8px;'>
                    Based on the provided metrics, the model estimates the student's final performance benchmark.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Big clean metric overlay
        st.metric(
            label="Estimated Final Score Outcome", 
            value=f"{prediction:.2f} / 100",
            delta=f"{prediction - previous_scores:+.2f} vs Previous" if prediction != previous_scores else None
        )
        
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction computation: {e}")
