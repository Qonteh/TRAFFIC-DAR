# Save this as traffic_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Traffic Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f4f6f9;
        padding: 20px;
    }
    
    /* Header styling */
    .header {
        background-color: #343a40;
        padding: 20px;
        color: white;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 0 1px rgba(0,0,0,.125), 0 1px 3px rgba(0,0,0,.2);
        margin-bottom: 20px;
        padding: 20px;
    }
    
    .card-header {
        border-bottom: 1px solid rgba(0,0,0,.125);
        padding-bottom: 10px;
        margin-bottom: 15px;
        font-size: 1.5rem;
        font-weight: 500;
        color: #343a40;
    }
    
    /* Info box */
    .info-box {
        background-color: #17a2b8;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    .success-box {
        background-color: #28a745;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    .warning-box {
        background-color: #ffc107;
        color: #343a40;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    .danger-box {
        background-color: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load models
@st.cache_resource
def load_models():
    models = {}
    model_files = [f for f in os.listdir('.') if f.endswith('_model.pkl')]
    
    if not model_files:
        st.error("No model files found. Please train models first.")
        return None
    
    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
        try:
            models[model_name] = joblib.load(model_file)
            print(f"Loaded {model_name}")
        except Exception as e:
            st.error(f"Error loading {model_name}: {e}")
    
    return models

# Function to load feature columns
@st.cache_data
def load_feature_columns():
    try:
        return joblib.load('feature_columns.pkl')
    except:
        st.error("Feature columns file not found. Please train models first.")
        return {'numerical': [], 'categorical': []}

# Function to load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv('dataset.csv')
    except:
        st.error("Dataset file not found. Please make sure 'dataset.csv' exists.")
        return None

# Function to make predictions
def predict_traffic(model, input_data):
    try:
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data
        
        # Make prediction
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Header
st.markdown('<div class="header"><h1 style="text-align: center;">Traffic Analysis Dashboard</h1></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div style="text-align: center;"><h2>Navigation</h2></div>', unsafe_allow_html=True)
    
    page = st.radio(
        "Select a page",
        ["Dashboard", "Model Comparison", "Prediction", "About"]
    )
    
    st.markdown("---")
    st.markdown('<div style="text-align: center;"><h3>About</h3></div>', unsafe_allow_html=True)
    st.info("""
    This dashboard provides insights into traffic patterns and allows you to predict traffic volume using machine learning.
    
    Five different models were trained and evaluated to find the best approach for traffic prediction.
    """)

# Load data and models
df = load_data()
models = load_models()
feature_columns = load_feature_columns()

# Main content based on selected page
if page == "Dashboard":
    st.markdown('<div class="card"><div class="card-header">Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    if df is not None:
        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.metric("Total Records", f"{df.shape[0]:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            # Assuming there's a traffic volume column
            traffic_col = [col for col in df.columns if any(term in col.lower() for term in ['volume', 'count', 'traffic'])]
            if traffic_col:
                avg_traffic = df[traffic_col[0]].mean()
                st.metric("Avg. Traffic Volume", f"{avg_traffic:.1f}")
            else:
                st.metric("Features", f"{df.shape[1]}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            if 'hour' in df.columns:
                peak_hour = df.groupby('hour').size().idxmax()
                st.metric("Peak Traffic Hour", f"{peak_hour}:00")
            else:
                st.metric("Data Types", f"{df.dtypes.nunique()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="danger-box">', unsafe_allow_html=True)
            if models is not None:
                st.metric("Models Available", f"{len(models)}")
            else:
                st.metric("Model Status", "Not Loaded")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key visualizations
    st.markdown('<div class="card"><div class="card-header">Key Insights</div>', unsafe_allow_html=True)
    
    # Display model comparison if available
    if os.path.exists('model_comparison.png'):
        st.image('model_comparison.png', caption="Model Performance Comparison")
        st.markdown("""
        **Model Comparison Description:**
        
        These charts compare the performance of five different machine learning models:
        - RMSE (Root Mean Squared Error): Lower values indicate better performance
        - MAE (Mean Absolute Error): Lower values indicate better performance
        - R² Score: Higher values indicate better performance
        - Accuracy (within 10%): Higher values indicate better performance
        - F2 Score: Higher values indicate better performance
        
        The best model is selected based on the highest R² score, which represents how well the model explains the variance in traffic volume.
        """)
    
    # Display feature importance if available
    if os.path.exists('feature_importance.png'):
        st.image('feature_importance.png', caption="Feature Importance")
        st.markdown("""
        **Feature Importance Description:**
        
        This chart shows which features have the most impact on traffic prediction:
        - Features are ranked from most important (top) to least important (bottom)
        - Longer bars indicate features with stronger influence on traffic
        - This helps identify the key factors affecting traffic patterns
        - Traffic management strategies can focus on these key factors
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Model Comparison":
    st.markdown('<div class="card"><div class="card-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    
    if models is not None:
        st.write(f"We trained and evaluated {len(models)} different machine learning models:")
        
        # List models
        for i, model_name in enumerate(models.keys(), 1):
            st.write(f"{i}. {model_name}")
        
        # Display model comparison image
        if os.path.exists('model_comparison.png'):
            st.image('model_comparison.png', caption="Model Performance Comparison")
        
        # Display individual model evaluations
        st.subheader("Individual Model Evaluations")
        
        for model_name in models.keys():
            image_path = f"{model_name.lower().replace(' ', '_')}_evaluation.png"
            if os.path.exists(image_path):
                st.write(f"### {model_name}")
                st.image(image_path, caption=f"{model_name} Evaluation")
            else:
                st.write(f"### {model_name}")
                st.warning(f"Evaluation visualization for {model_name} not found.")
    else:
        st.error("No models loaded. Please train models first.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Prediction":
    st.markdown('<div class="card"><div class="card-header">Traffic Prediction</div>', unsafe_allow_html=True)
    
    if models is not None:
        # Model selection
        model_names = list(models.keys())
        selected_model = st.selectbox("Select a model for prediction", model_names)
        model = models[selected_model]
        
        st.write(f"Using {selected_model} for prediction")
        
        # Create input form
        st.subheader("Enter Traffic Parameters")
        
        # Get feature columns
        numerical_cols = feature_columns['numerical']
        categorical_cols = feature_columns['categorical']
        
        # Create input form
        input_data = {}
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        # Numerical features
        with col1:
            st.write("### Numerical Features")
            for feature in numerical_cols:
                # Try to get min and max values from the dataset
                if df is not None and feature in df.columns:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    default_val = float(df[feature].mean())
                else:
                    min_val = 0.0
                    max_val = 100.0
                    default_val = 50.0
                
                input_data[feature] = st.slider(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100
                )
        
        # Categorical features
        with col2:
            st.write("### Categorical Features")
            for feature in categorical_cols:
                # Get unique values from dataset
                if df is not None and feature in df.columns:
                    unique_values = df[feature].unique().tolist()
                    default_value = unique_values[0] if unique_values else ""
                else:
                    unique_values = ["Option 1", "Option 2", "Option 3"]
                    default_value = "Option 1"
                
                input_data[feature] = st.selectbox(
                    f"{feature}",
                    options=unique_values,
                    index=0
                )
        
        # Predict button
        if st.button("Predict Traffic"):
            prediction = predict_traffic(model, input_data)
            
            if prediction is not None:
                st.markdown('<div style="background-color: #28a745; padding: 20px; border-radius: 5px; color: white; text-align: center; font-size: 24px; margin-top: 20px;">', unsafe_allow_html=True)
                st.markdown(f"<h2>Predicted Traffic Volume: {prediction:.2f}</h2>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Interpretation
                st.subheader("Prediction Interpretation")
                
                # Try to determine if the prediction is high, medium, or low
                if df is not None:
                    target_cols = [col for col in df.columns if any(term in col.lower() for term in ['volume', 'count', 'traffic'])]
                    if target_cols:
                        target_col = target_cols[0]
                        low_threshold = df[target_col].quantile(0.33)
                        high_threshold = df[target_col].quantile(0.66)
                        
                        if prediction < low_threshold:
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown("**Low Traffic Volume**", unsafe_allow_html=True)
                            st.write("The predicted traffic volume is relatively low compared to historical data.")
                            st.write("This suggests favorable traffic conditions with minimal congestion.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif prediction > high_threshold:
                            st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                            st.markdown("**High Traffic Volume**", unsafe_allow_html=True)
                            st.write("The predicted traffic volume is relatively high compared to historical data.")
                            st.write("This suggests potential congestion and delays. Consider alternative routes or travel times.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.markdown("**Moderate Traffic Volume**", unsafe_allow_html=True)
                            st.write("The predicted traffic volume is moderate compared to historical data.")
                            st.write("Some congestion may be expected, but severe delays are unlikely.")
                            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("No models loaded. Please train models first.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "About":
    st.markdown('<div class="card"><div class="card-header">About This Project</div>', unsafe_allow_html=True)
    
    st.write("""
    ## Traffic Analysis and Prediction
    
    This project aims to analyze traffic patterns and predict traffic volume using machine learning techniques.
    
    ### Models Used
    
    We trained and evaluated five different machine learning models:
    
    1. **Linear Regression**: A simple model that assumes a linear relationship between features and traffic volume.
    2. **Random Forest**: An ensemble method that combines multiple decision trees to improve prediction accuracy.
    3. **Gradient Boosting**: An ensemble technique that builds trees sequentially, with each tree correcting errors of previous trees.
    4. **AdaBoost**: An adaptive boosting algorithm that gives more weight to difficult-to-predict instances.
    5. **K-Nearest Neighbors**: A non-parametric method that predicts based on the most similar training examples.
    
    ### Evaluation Metrics
    
    The models were evaluated using several metrics:
    
    - **RMSE (Root Mean Squared Error)**: Measures the average magnitude of prediction errors.
    - **MAE (Mean Absolute Error)**: Measures the average absolute difference between predicted and actual values.
    - **R² Score**: Indicates how well the model explains the variance in traffic volume.
    - **Accuracy (within 10%)**: Percentage of predictions within 10% of actual values.
    - **F2 Score**: A metric that gives more weight to recall than precision.
    
    ### How to Use This Dashboard
    
    - **Dashboard**: View key metrics and visualizations about traffic patterns.
    - **Model Comparison**: Compare the performance of different machine learning models.
    - **Prediction**: Use the trained models to predict traffic volume based on input parameters.
    
    ### Data Sources
    
    The models were trained on historical traffic data, including factors such as:
    
    - Time of day
    - Day of week
    - Weather conditions
    - Location
    - Road conditions
    - Special events
    
    ### Future Improvements
    
    - Incorporate real-time data feeds
    - Add more advanced models like neural networks
    - Develop a mobile app for on-the-go predictions
    - Integrate with navigation systems
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="background-color: #343a40; padding: 10px; border-radius: 5px; color: white; text-align: center; margin-top: 20px;">
    <p>Traffic Analysis Dashboard © 2025</p>
</div>
""", unsafe_allow_html=True)