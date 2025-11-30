# [file name]: training.py
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import tempfile
from datetime import datetime
import sys
from project_paths import get_model_paths

def setup_mlflow_training():
    """Setup and display MLflow training interface"""
    st.title("🔬 MLflow Model Training & Tracking")
    
    # MLflow setup information
    st.subheader("📊 MLflow Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Tracking URI:**\n`{mlflow.get_tracking_uri()}`")
    
    with col2:
        artifact_root = os.environ.get('MLFLOW_ARTIFACT_ROOT', './mlruns')
        st.info(f"**Artifact Root:**\n`{artifact_root}`")
    
    # Training section
    st.subheader("🚀 Model Training")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Prophet", "ARIMA", "LightGBM"],
        help="Choose which model to train"
    )
    
    # Training parameters
    st.subheader("⚙️ Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data to use for testing"
        )
    
    with col2:
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=2,
            max_value=5,
            value=3,
            help="Number of cross-validation folds"
        )
    
    # Start training button
    if st.button("🎯 Start Model Training", type="primary"):
        with st.spinner(f"Training {model_type} model..."):
            try:
                result = train_model(model_type, test_size, cv_folds)
                if result:
                    st.success("✅ Model training completed successfully!")
                    st.balloons()
                else:
                    st.error("❌ Model training failed!")
            except Exception as e:
                st.error(f"❌ Training error: {e}")
    
    # Display existing runs
    st.subheader("📈 Existing Training Runs")
    display_existing_runs()
    
    # Model registry info
    st.subheader("🏷️ Model Registry")
    display_model_registry()

# In training.py, update the train_model function:

def train_model(model_type, test_size, cv_folds):
    """Train a model and log to MLflow"""
    try:
        paths = get_model_paths()
        
        # Load dataset
        if not os.path.exists(paths['dataset']):
            st.error(f"Dataset not found at {paths['dataset']}")
            return False
        
        # Load and prepare data
        df = pd.read_csv(paths['dataset'])
        st.info(f"📊 Loaded dataset with {len(df)} rows")
        
        # Set up MLflow experiment
        experiment_name = f"{model_type}_Training"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                "model_type": model_type.lower(),
                "test_size": test_size,
                "cv_folds": cv_folds,
                "dataset_size": len(df),
                "training_time": datetime.now().isoformat()
            })
            
            # Simulate training (replace with actual model training)
            st.info(f"🤖 Training {model_type} model...")
            
            # Create appropriate model based on type
            if model_type.lower() == "lightgbm":
                import lightgbm as lgb
                from model_wrappers import LightGBMModelWrapper
                
                # Create a simple LightGBM model
                model = lgb.LGBMRegressor()
                # Here you would normally train the model with your data
                # For demo, we'll use a dummy model
                wrapper = LightGBMModelWrapper(model)
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=wrapper,
                    registered_model_name="BestForecastModels"
                )
                
            elif model_type.lower() == "prophet":
                from model_wrappers import ProphetModelWrapper
                from prophet import Prophet
                
                # Create Prophet model
                model = Prophet()
                wrapper = ProphetModelWrapper(model)
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=wrapper,
                    registered_model_name="BestForecastModels"
                )
                
            else:  # ARIMA
                from model_wrappers import ARIMAModelWrapper
                from statsmodels.tsa.arima.model import ARIMA
                
                # Create ARIMA model (dummy for demo)
                wrapper = ARIMAModelWrapper(None)
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=wrapper,
                    registered_model_name="BestForecastModels"
                )
            
            # Simulate metrics based on model type
            if model_type.lower() == "prophet":
                metrics = {
                    "mae": np.random.uniform(10000, 50000),
                    "rmse": np.random.uniform(15000, 60000),
                    "r2_score": np.random.uniform(0.7, 0.95),
                    "mape": np.random.uniform(0.05, 0.15)
                }
            elif model_type.lower() == "arima":
                metrics = {
                    "mae": np.random.uniform(12000, 55000),
                    "rmse": np.random.uniform(18000, 65000),
                    "r2_score": np.random.uniform(0.65, 0.9),
                    "mape": np.random.uniform(0.08, 0.18)
                }
            else:  # lightgbm
                metrics = {
                    "mae": np.random.uniform(8000, 45000),
                    "rmse": np.random.uniform(12000, 50000),
                    "r2_score": np.random.uniform(0.75, 0.98),
                    "mape": np.random.uniform(0.04, 0.12)
                }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Set tags
            mlflow.set_tags({
                "framework": "mlflow",
                "task": "forecasting",
                "author": "streamlit_app"
            })
            
            # Log training summary as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Training Summary - {model_type}\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Dataset: {len(df)} rows\n")
                f.write(f"Test Size: {test_size}%\n")
                f.write(f"CV Folds: {cv_folds}\n")
                f.write("\nMetrics:\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
                mlflow.log_artifact(f.name, "training_summary")
            
            st.success(f"✅ {model_type} training completed!")
            st.metric("MAE", f"${metrics['mae']:,.0f}")
            st.metric("R² Score", f"{metrics['r2_score']:.3f}")
            
            return True
            
    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return False

def create_dummy_model():
    """Create a dummy model for demonstration"""
    from sklearn.dummy import DummyRegressor
    return DummyRegressor(strategy="mean")

def display_existing_runs():
    """Display existing MLflow runs"""
    try:
        client = MlflowClient()
        
        # Get all experiments
        experiments = client.search_experiments()
        
        for exp in experiments:
            with st.expander(f"📁 Experiment: {exp.name}", expanded=False):
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=5
                )
                
                for run in runs:
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.write(f"**Run:** {run.info.run_name}")
                        st.write(f"Status: {run.info.status}")
                    
                    with col2:
                        st.write(f"Start: {pd.to_datetime(run.info.start_time).strftime('%Y-%m-%d %H:%M')}")
                        if 'mae' in run.data.metrics:
                            st.write(f"MAE: ${run.data.metrics['mae']:,.0f}")
                    
                    with col3:
                        if st.button("View", key=run.info.run_id):
                            display_run_details(run)
    
    except Exception as e:
        st.warning(f"Could not load runs: {e}")

def display_run_details(run):
    """Display detailed information about a run"""
    st.subheader(f"Run Details: {run.info.run_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Parameters:**")
        for key, value in run.data.params.items():
            st.write(f"- {key}: {value}")
    
    with col2:
        st.write("**Metrics:**")
        for key, value in run.data.metrics.items():
            st.write(f"- {key}: {value:.4f}")
    
    st.write("**Tags:**")
    for key, value in run.data.tags.items():
        st.write(f"- {key}: {value}")

def display_model_registry():
    """Display model registry information"""
    try:
        client = MlflowClient()
        
        # Get registered models
        models = client.search_registered_models()
        
        if not models:
            st.info("No models registered yet. Train a model to see it here!")
            return
        
        for model in models:
            with st.expander(f"🏷️ Model: {model.name}", expanded=True):
                for version in model.latest_versions:
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"Version {version.version}")
                    
                    with col2:
                        st.write(f"Stage: **{version.current_stage}**")
                    
                    with col3:
                        st.write(f"Created: {pd.to_datetime(version.creation_timestamp).strftime('%Y-%m-%d')}")
                    
                    with col4:
                        if version.current_stage == "None":
                            if st.button("Promote", key=f"promote_{version.version}"):
                                promote_model_version(model.name, version.version)
    
    except Exception as e:
        st.warning(f"Could not load model registry: {e}")

def promote_model_version(model_name, version):
    """Promote a model version to staging"""
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        st.success(f"✅ Version {version} promoted to Staging!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to promote model: {e}")

def reset_mlflow_completely():
    """Completely reset MLflow (use with caution)"""
    try:
        import shutil
        
        # Reset MLflow tracking
        mlflow.set_tracking_uri("./mlruns")
        
        # Delete mlruns directory
        if os.path.exists("./mlruns"):
            shutil.rmtree("./mlruns")
            st.success("🗑️ Deleted mlruns directory")
        
        # Delete MLflow database
        if os.path.exists("mlflow.db"):
            os.remove("mlflow.db")
            st.success("🗑️ Deleted mlflow.db")
        
        # Recreate directories
        os.makedirs("./mlruns", exist_ok=True)
        st.success("📁 Recreated mlruns directory")
        
        # Reset session state
        if 'model' in st.session_state:
            del st.session_state['model']
        
        st.success("🔄 MLflow reset complete! You can now retrain models.")
        
    except Exception as e:
        st.error(f"❌ Error resetting MLflow: {e}")

