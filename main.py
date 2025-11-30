import streamlit as st
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure MLflow to use relative paths before importing MLFlow
os.makedirs('mlruns', exist_ok=True)
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
os.environ['MLFLOW_ARTIFACT_ROOT'] = 'file:./mlruns'

# Now import other modules
try:
    from data_loader import load_data
    from model_registry import load_production_model_from_registry, get_model_type_from_registry
    from dashboard import run_dashboard
    from forecast_ui import run_forecast_app
    from training import setup_mlflow_training, reset_mlflow_completely
    from monitoring import run_monitoring_app
    from model_registry import fix_mlflow_paths, recreate_model_registry as recreate_registry
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# MLflow Setup - Import after environment variables are set
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    layout="wide", 
    page_title="Data Analysis & Forecast App", 
    page_icon="📊"
)

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def check_mlflow_ready():
    """Ensure MLflow is ready for the session"""
    try:
        client = MlflowClient()
        client.list_experiments()
        return True
    except:
        return False

def setup_mlflow_for_session():
    """Initialize MLflow for new session"""
    try:
        # Create necessary directories
        os.makedirs('mlruns', exist_ok=True)
        
        # Set up default experiment
        mlflow.set_experiment("best_models")
        
        st.success("✅ MLflow ready for this session")
        return True
    except Exception as e:
        st.error(f"❌ MLflow setup failed: {e}")
        return False

def show_session_info():
    """Display session information"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕒 Session Info")
    
    # Show MLflow status
    if check_mlflow_ready():
        st.sidebar.success("MLflow: Active")
        
        # Count trained models in this session
        try:
            client = MlflowClient()
            experiments = client.list_experiments()
            run_count = 0
            for exp in experiments:
                runs = client.search_runs([exp.experiment_id])
                run_count += len(runs)
            
            st.sidebar.info(f"Models this session: {run_count}")
        except:
            st.sidebar.info("Models this session: 0")
    else:
        st.sidebar.warning("MLflow: Not Ready")
        if st.sidebar.button("🔄 Initialize MLflow for This Session"):
            if setup_mlflow_for_session():
                st.rerun()

def setup_mlflow_utilities():
    """Setup MLflow utility functions in sidebar"""
    st.sidebar.subheader("🔧 MLflow Utilities")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🛠️ Fix Paths", help="Check and fix MLflow model paths"):
            try:
                fix_mlflow_paths()
            except Exception as e:
                st.sidebar.error(f"Error fixing paths: {e}")
    
    with col2:
        if st.button("🔄 Recreate Registry", help="Recreate model registry from existing runs"):
            try:
                recreate_registry()
            except Exception as e:
                st.sidebar.error(f"Error recreating registry: {e}")
    
    if st.sidebar.button("🗑️ Clear Session Models", type="secondary", help="Clear all models from this session"):
        try:
            reset_mlflow_completely()
            st.sidebar.success("Session models cleared! Please refresh the page.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error clearing models: {e}")
    
    st.sidebar.markdown("---")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    # Initialize Session State
    defaults = {
        'forecast_data': None,
        'model_type': None,
        'real_time_predictions': [],
        'forecast_end_date': None,
        'forecast_periods': None,
        'mlflow_initialized': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Load resources
    train, min_date, max_date, sort_state, prophet_df = load_data()
    
    # Load production model from MLflow Registry
    model = load_production_model_from_registry()
    model_type = get_model_type_from_registry() if model else "unknown"
    st.session_state.model_type = model_type

    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    app_mode = st.sidebar.selectbox(
        "Go to", 
        ["📊 Dashboard", "🚀 Forecast Engine", "📈 Monitoring", "🔬 MLflow Tracking"],
        key="app_mode"
    )
    
    # Show session info
    show_session_info()
    
    # Add MLflow utilities to sidebar for MLflow Tracking page
    if "MLflow Tracking" in app_mode:
        setup_mlflow_utilities()

    # Check if MLflow is ready for MLflow-related pages
    if app_mode in ["🚀 Forecast Engine", "📈 Monitoring", "🔬 MLflow Tracking"]:
        if not check_mlflow_ready():
            st.warning("🔧 MLflow not initialized for this session")
            if st.button("🔄 Initialize MLflow for This Session"):
                if setup_mlflow_for_session():
                    st.rerun()
            return

    # Main application routing
    if "Dashboard" in app_mode:
        run_dashboard(train, min_date, max_date, sort_state)
    elif "Forecast Engine" in app_mode:
        run_forecast_app(model, prophet_df, model_type)
    elif "Monitoring" in app_mode:
        run_monitoring_app()
    else:  # MLflow Tracking
        setup_mlflow_training()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Session Status:**\n"
        f"- MLflow Ready: `{'Yes' if check_mlflow_ready() else 'No'}`\n"
        f"- Production Model: `{model_type if model else 'None'}`\n"
        "**Note:** Models persist only during this browser session"
    )

if __name__ == '__main__':
    main()