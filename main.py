# [file name]: main.py
import streamlit as st
import os
import sys
import pandas as pd

# Set the working directory to the project root
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Configure MLflow to use file store instead of SQLite for Streamlit Cloud compatibility
os.environ['MLFLOW_ARTIFACT_ROOT'] = './mlruns'
os.environ['MLFLOW_TRACKING_URI'] = './mlruns'  # Use file store instead of SQLite

# Create necessary directories
os.makedirs('./mlruns', exist_ok=True)

# Now import other modules
from data_loader import load_data
from model_registry import load_production_model_from_registry, get_model_type_from_registry
from dashboard import run_dashboard
from forecast_ui import run_forecast_app
from training import setup_mlflow_training, reset_mlflow_completely
from monitoring import run_monitoring_app
from model_registry import fix_mlflow_paths, recreate_model_registry as recreate_registry

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    layout="wide", 
    page_title="Data Analysis & Forecast App", 
    page_icon="📊"
)

# MLflow Setup - Import after environment variables are set
import mlflow
from mlflow.tracking import MlflowClient

# Use file store instead of SQLite for Streamlit Cloud compatibility
mlflow.set_tracking_uri("./mlruns")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# In main.py, add this function and update the MLflow utilities section:

def setup_mlflow_utilities():
    """Setup MLflow utility functions in sidebar"""
    st.sidebar.subheader("🔧 MLflow Utilities")
    
    # Add registry status check
    if st.sidebar.button("📊 Check Registry Status"):
        from model_registry import check_registry_status
        check_registry_status()
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🛠️ Fix Paths", help="Check and fix MLflow model paths"):
            try:
                from model_registry import fix_mlflow_paths
                fix_mlflow_paths()
            except Exception as e:
                st.sidebar.error(f"Error fixing paths: {e}")
    
    with col2:
        if st.button("🔄 Recreate Registry", help="Recreate model registry from existing runs"):
            try:
                from model_registry import recreate_model_registry
                recreate_model_registry()
            except Exception as e:
                st.sidebar.error(f"Error recreating registry: {e}")
    
    if st.sidebar.button("🗑️ Reset MLflow", type="secondary", help="Completely reset MLflow (requires retraining)"):
        try:
            from training import reset_mlflow_completely
            reset_mlflow_completely()
            st.sidebar.success("MLflow reset complete! Please refresh the page.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error resetting MLflow: {e}")
    
    st.sidebar.markdown("---")

def main():
    """Main application entry point"""
    # Initialize Session State
    defaults = {
        'forecast_data': None,
        'model_type': None,
        'real_time_predictions': [],
        'forecast_end_date': None,
        'forecast_periods': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Load resources with error handling
    try:
        train, min_date, max_date, sort_state, prophet_df = load_data()
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        train, min_date, max_date, sort_state, prophet_df = pd.DataFrame(), None, None, pd.Series(), pd.DataFrame()
    
    # Load production model from MLflow Registry with better error handling
    try:
        model = load_production_model_from_registry()
        model_type = get_model_type_from_registry() if model else "unknown"
    except Exception as e:
        st.sidebar.warning("No trained models found. Please train models first.")
        model = None
        model_type = "unknown"

    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    app_mode = st.sidebar.selectbox(
        "Go to", 
        ["📊 Dashboard", "🚀 Forecast Engine", "📈 Monitoring", "🔬 MLflow Tracking"],
        key="app_mode"
    )
    
    # Add data source info to sidebar
    st.sidebar.markdown("---")
    st.sidebar.info("**Data Sources:**\n"
                   "- Dashboard: Data.csv from Data.zip\n"
                   "- Training: model_dataset.csv")
    
    # Add MLflow utilities to sidebar for MLflow Tracking page
    if "MLflow Tracking" in app_mode:
        setup_mlflow_utilities()

    # Main application routing - FIXED THE TYPO HERE
    if "Dashboard" in app_mode:
        run_dashboard(train, min_date, max_date, sort_state)
    elif "Forecast Engine" in app_mode:  # FIXED: appmode -> app_mode
        run_forecast_app(model, prophet_df, model_type)
    elif "Monitoring" in app_mode:
        run_monitoring_app()
    else:  # MLflow Tracking
        setup_mlflow_training()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**MLflow Status:**\n"
        f"- Tracking URI: `{mlflow.get_tracking_uri()}`\n"
        f"- Artifact Root: `{os.environ.get('MLFLOW_ARTIFACT_ROOT', 'Not set')}`\n"
        f"- Production Model: `{model_type if model else 'None'}`"
    )

if __name__ == '__main__':
    main()

