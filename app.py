import streamlit as st
import pickle
import pandas as pd

# Set up page configurations
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load the trained KNN model safely
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Title and description
st.title("🎓 Student Performance Predictor")
st.write("Enter the student metrics below to estimate their final performance score.")
st.markdown("---")

# Layout columns for UI grouping
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider(
        "Hours Studied (per week)", 
        min_value=0.0, 
        max_value=100.0, 
        value=10.0, 
        step=0.5,
        help="Total hours spent studying per week."
    )
    
    attendance_percent = st.slider(
        "Attendance Percentage (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=85.0, 
        step=1.0,
        help="Overall class attendance rate."
    )

with col2:
    sleep_hours = st.slider(
        "Sleep Hours (per night)", 
        min_value=0.0, 
        max_value=14.0, 
        value=7.0, 
        step=0.5,
        help="Average hours of sleep per night."
    )
    
    previous_scores = st.number_input(
        "Previous Exam Score", 
        min_value=0.0, 
        max_value=100.0, 
        value=75.0, 
        step=1.0,
        help="The student's score on their last examination."
    )

st.markdown("---")

# Predict action
if st.button("Predict Score", type="primary"):
    # CRITICAL: Constructing a DataFrame preserves the exact feature names 
    # ('hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores')
    # that your model was trained on, preventing mismatch warnings.
    input_df = pd.DataFrame([{
        'hours_studied': hours_studied,
        'sleep_hours': sleep_hours,
        'attendance_percent': attendance_percent,
        'previous_scores': previous_scores
    }])
    
    try:
        # Run prediction
        prediction = model.predict(input_df)[0]
        
        # Display Results
        st.success("### Prediction Complete!")
        st.metric(label="Predicted Final Score", value=f"{prediction:.2f} / 100")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
