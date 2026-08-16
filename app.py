import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# -------------------------------------------------------------
# 1. Page Configuration & Layout
# -------------------------------------------------------------
st.set_page_config(
    page_title="Employee Churn Analytics & Model Studio",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #4F46E5, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .subtitle {
        color: #6B7280;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #1E293B;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.3rem;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3B82F6;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Model Backup and Management
# -------------------------------------------------------------
MODEL_PATH = "model_and_key_components.pkl"
DEFAULT_MODEL_PATH = "model_and_key_components_default.pkl"

# Create a backup of the baseline model on first run if it doesn't exist
if not os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(MODEL_PATH):
    shutil.copy(MODEL_PATH, DEFAULT_MODEL_PATH)

def load_model_components():
    """Loads the model and unique values dictionary from disk."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    else:
        st.error("Model file not found! Please check that `model_and_key_components.pkl` exists in your workspace.")
        return None

# Load model data into session state to allow dynamic updates without restarting
if 'model_data' not in st.session_state:
    st.session_state['model_data'] = load_model_components()

# Extract source from loaded data dynamically
if 'model_source' not in st.session_state:
    model_data = st.session_state.get('model_data')
    if isinstance(model_data, dict):
        st.session_state['model_source'] = model_data.get('source', "Default Baseline")
    else:
        st.session_state['model_source'] = "Default Baseline"

# Helper to reload model in-memory
def reload_model_data():
    st.session_state['model_data'] = load_model_components()
    model_data = st.session_state['model_data']
    if isinstance(model_data, dict):
        st.session_state['model_source'] = model_data.get('source', "Default Baseline")
    else:
        st.session_state['model_source'] = "Default Baseline"

# -------------------------------------------------------------
# 3. Model Training Functions
# -------------------------------------------------------------
def clean_attrition_target(series):
    """Normalizes the target series to binary 0 and 1."""
    cleaned = series.copy()
    if cleaned.dtype == 'object' or cleaned.dtype == 'str':
        cleaned = cleaned.astype(str).str.strip().str.lower()
        cleaned = cleaned.map({'true': 1, 'yes': 1, 'y': 1, '1': 1, 'false': 0, 'no': 0, 'n': 0, '0': 0})
    else:
        cleaned = cleaned.astype(int)
    return cleaned.fillna(0).astype(int)

def train_model_from_df(raw_df):
    """Trains a new CatBoost model based on the uploaded DataFrame."""
    required_cols = [
        'Age', 'Department', 'EnvironmentSatisfaction', 'JobRole',
        'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime',
        'PercentSalaryHike', 'RelationshipSatisfaction',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsSinceLastPromotion',
        'YearsWithCurrManager', 'Attrition'
    ]
    
    # Check for missing columns
    missing = [col for col in required_cols if col not in raw_df.columns]
    if missing:
        raise ValueError(f"The uploaded CSV is missing the following required columns: {', '.join(missing)}")
    
    # Select only required columns and create copy
    df = raw_df[required_cols].copy()
    
    # Clean target
    df['Attrition'] = clean_attrition_target(df['Attrition'])
    
    # Extract target and features
    target_df = pd.DataFrame(df['Attrition'])
    features_df = df.drop('Attrition', axis=1)
    
    # Rebalance dataset (Oversample minority class)
    oversampler = RandomOverSampler(random_state=42)
    features_balanced, target_balanced = oversampler.fit_resample(features_df, target_df)
    
    X = features_balanced
    y = target_balanced['Attrition']
    
    # Train-test split
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get categorical features indices (supporting object, category, string, str types)
    cat_cols_index = [i for i, col in enumerate(X.columns) if X[col].dtype.name in ['object', 'category', 'string', 'str']]
    
    # Convert categorical features to string datatype
    for i in range(len(X_train.columns)):
        if i in cat_cols_index:
            col_name = X_train.columns[i]
            X_train[col_name] = X_train[col_name].astype(str)
            X_eval[col_name] = X_eval[col_name].astype(str)
            
    # Initialize and fit CatBoostClassifier
    model = CatBoostClassifier(random_state=42, n_estimators=50, verbose=0)
    trained_model = model.fit(X_train, y_train, cat_features=cat_cols_index)
    
    # Calculate performance metrics
    y_pred = trained_model.predict(X_eval)
    accuracy = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred, average='binary')
    
    # Extract unique values for dynamic selectboxes
    unique_values = {}
    for column in X.columns:
        unique_values[column] = X[column].unique()
        
    return trained_model, unique_values, accuracy, f1

# -------------------------------------------------------------
# 4. User Interface Header
# -------------------------------------------------------------
st.markdown('<div class="main-title">Employee Churn Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict employee attrition and retrain ML models dynamically</div>', unsafe_allow_html=True)

# Tabs navigation
tab_predict, tab_studio = st.tabs(["🔮 Single Employee Prediction", "⚙️ Model Studio & Upload"])

# -------------------------------------------------------------
# Tab 1: Single Prediction Interface
# -------------------------------------------------------------
with tab_predict:
    if st.session_state['model_data'] is not None:
        model = st.session_state['model_data']['model']
        unique_values = st.session_state['model_data'].get('unique_values', {})
        
        # Display active model source
        st.info(f"🚀 **Active Model:** {st.session_state['model_source']}")
        
        # Safe default listings if unique_values is empty or missing keys
        dept_options = list(unique_values.get("Department", ["Sales", "Research & Development", "Human Resources"]))
        job_options = list(unique_values.get("JobRole", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ]))
        
        # Multi-column input layout for modern aesthetics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Demographics & Career</div>', unsafe_allow_html=True)
            age = st.slider("Age", 18, 65, 30, help="Employee age in years")
            department = st.selectbox("Department", dept_options, help="Current department of the employee")
            job_role = st.selectbox("Job Role", job_options, help="Detailed designation within the department")
            monthly_income = st.number_input("Monthly Income ($)", 1000, 500000, 5000, step=500, help="Monthly base salary")
            
            st.markdown('<div class="section-header">Traction & Tenancy</div>', unsafe_allow_html=True)
            num_companies = st.slider("Number of Companies Worked", 0, 10, 2, help="Number of previous companies worked at")
            percent_hike = st.slider("Percent Salary Hike (%)", 10, 30, 15, help="Most recent salary hike percentage")
            training_times = st.slider("Training Times Last Year", 0, 10, 2, help="Number of training sessions attended last year")
            
        with col2:
            st.markdown('<div class="section-header">Work Satisfaction & Environment</div>', unsafe_allow_html=True)
            env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3, help="1 = Low, 4 = High")
            job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3, help="1 = Low, 4 = High")
            rel_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3, help="With colleagues: 1 = Low, 4 = High")
            work_life = st.slider("Work Life Balance", 1, 4, 3, help="1 = Bad, 4 = Excellent")
            
            st.markdown('<div class="section-header">Condition & Tenancy (Contd.)</div>', unsafe_allow_html=True)
            over_time = st.checkbox("Over Time", value=False, help="Does the employee work overtime regularly?")
            promotion = st.slider("Years Since Last Promotion", 0, 15, 2, help="Years passed since the last promotion")
            manager = st.slider("Years With Current Manager", 0, 20, 3, help="Tenure under current manager in years")
            
        # Prediction Output Area
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict Attrition", type="primary", use_container_width=True):
            input_df = pd.DataFrame({
                "Age": [age],
                "Department": [department],
                "EnvironmentSatisfaction": [env_satisfaction],
                "JobRole": [job_role],
                "JobSatisfaction": [job_satisfaction],
                "MonthlyIncome": [monthly_income],
                "NumCompaniesWorked": [num_companies],
                "OverTime": [over_time],
                "PercentSalaryHike": [percent_hike],
                "RelationshipSatisfaction": [rel_satisfaction],
                "TrainingTimesLastYear": [training_times],
                "WorkLifeBalance": [work_life],
                "YearsSinceLastPromotion": [promotion],
                "YearsWithCurrManager": [manager]
            })
            
            # Predict
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            st.markdown("---")
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                st.subheader("Analysis & Outcome")
                if prediction == 1 or probability > 0.5:
                    st.error("### ⚠️ High Risk of Churn")
                    st.write("This employee is likely to leave the organization. Consider intervention strategies such as compensation adjustments, career development discussions, or improving team engagement.")
                else:
                    st.success("### ✅ Low Risk of Churn")
                    st.write("This employee is highly likely to stay with the organization. Continue supporting their growth and environment satisfaction.")
            
            with res_col2:
                st.subheader("Probability Score")
                st.metric(label="Attrition Probability", value=f"{probability * 100:.2f}%")
                st.progress(probability)
                
    else:
        st.warning("No model found. Please load or train a model in the Model Studio tab.")

# -------------------------------------------------------------
# Tab 2: Model Studio & Training
# -------------------------------------------------------------
with tab_studio:
    st.markdown('<div class="section-header">Upload Training Data & Update Model</div>', unsafe_allow_html=True)
    st.write(
        "Upload new employee datasets to retrain the churn prediction model. "
        "The file must be in CSV format and include all 15 core parameters (14 features and the `Attrition` target column)."
    )
    
    # Template download option
    template_file = "train_data.csv"
    if os.path.exists(template_file):
        with open(template_file, "r") as f:
            template_csv = f.read()
        st.download_button(
            label="📥 Download Template CSV",
            data=template_csv,
            file_name="employee_churn_template.csv",
            mime="text/csv",
            help="Download the default dataset structure to use as a format template."
        )
    else:
        st.info("Template file (train_data.csv) not found on disk, but you can upload any CSV containing: Age, Department, EnvironmentSatisfaction, JobRole, JobSatisfaction, MonthlyIncome, NumCompaniesWorked, OverTime, PercentSalaryHike, RelationshipSatisfaction, TrainingTimesLastYear, WorkLifeBalance, YearsSinceLastPromotion, YearsWithCurrManager, and Attrition.")
        
    st.markdown("---")
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose a CSV file for Retraining", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read uploaded CSV
            raw_df = pd.read_csv(uploaded_file)
            st.success("CSV file successfully uploaded!")
            
            # Show preview
            st.subheader("Uploaded Dataset Preview")
            st.dataframe(raw_df.head(5))
            
            # Trigger Training
            if st.button("🚀 Train Custom Model", type="primary", use_container_width=True):
                with st.spinner("Retraining model in progress... Preprocessing data, balancing target classes, and fitting CatBoost Classifier..."):
                    try:
                        trained_model, unique_values, accuracy, f1 = train_model_from_df(raw_df)
                        
                        # Save components to disk (including source metadata)
                        saved_components = {
                            'model': trained_model,
                            'unique_values': unique_values,
                            'source': "Custom Trained (Uploaded Data)"
                        }
                        with open(MODEL_PATH, "wb") as file:
                            pickle.dump(saved_components, file)
                        
                        # Update session state model
                        st.session_state['model_data'] = saved_components
                        st.session_state['model_source'] = "Custom Trained (Uploaded Data)"
                        
                        st.balloons()
                        st.success("Model successfully retrained and updated on disk!")
                        
                        # Show Metrics Cards
                        st.markdown('<div class="section-header">New Model Performance Metrics</div>', unsafe_allow_html=True)
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{accuracy * 100:.2f}%</div>
                                <div class="metric-label">Evaluation Accuracy</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m_col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{f1 * 100:.2f}%</div>
                                <div class="metric-label">Evaluation F1-Score</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m_col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{len(raw_df):,}</div>
                                <div class="metric-label">Training Dataset Size</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    except Exception as train_error:
                        st.error(f"Error during training: {train_error}")
                        
        except Exception as read_error:
            st.error(f"Error reading CSV file: {read_error}")
            
    # Reset/Restore model settings
    st.markdown("---")
    st.markdown('<div class="section-header">Baseline Model Recovery</div>', unsafe_allow_html=True)
    st.write("Reset the application model to the default baseline model trained on the standard HR attrition dataset.")
    
    if st.button("🔄 Reset to Default Model", use_container_width=True):
        if os.path.exists(DEFAULT_MODEL_PATH):
            shutil.copy(DEFAULT_MODEL_PATH, MODEL_PATH)
            reload_model_data()
            st.session_state['model_source'] = "Default Baseline"
            st.success("Restored the baseline model successfully!")
            st.rerun()
        else:
            st.error("Default baseline backup model (model_and_key_components_default.pkl) not found on disk.")