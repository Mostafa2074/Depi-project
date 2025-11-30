# [file name]: model_registry.py
import streamlit as st
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os
import tempfile
import shutil
import pandas as pd

@st.cache_resource(show_spinner=False)
def load_production_model_from_registry(model_name="BestForecastModels", stage="Production"):
    """Loads the production model from MLflow Model Registry with robust path handling."""
    try:
        client = MlflowClient()
        
        # Clear cache to ensure we get fresh data
        st.cache_resource.clear()
        
        st.sidebar.info("🔄 Checking for production models...")
        
        # Get all model versions
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not model_versions:
            st.sidebar.warning(f"❌ No models found in registry for {model_name}")
            st.sidebar.info("💡 Please train models in the MLflow Tracking tab first.")
            return None
        
        # Debug: Show all available models and their stages
        st.sidebar.info(f"📋 Found {len(model_versions)} model versions:")
        for mv in model_versions:
            st.sidebar.write(f"  - Version {mv.version}: {mv.current_stage}")
            
        # Get production models
        production_models = [mv for mv in model_versions if mv.current_stage == stage]
        
        if not production_models:
            st.sidebar.warning(f"❌ No model found in {stage} stage for {model_name}")
            st.sidebar.info("💡 Please promote a model to Production stage in the MLflow Tracking tab.")
            
            # Show available stages for user guidance
            available_stages = list(set(mv.current_stage for mv in model_versions))
            st.sidebar.info(f"Available stages: {', '.join(available_stages)}")
            return None
        
        # Get the latest production model
        latest_production = max(production_models, key=lambda x: int(x.version))
        model_uri = f"models:/{model_name}/{latest_production.version}"
        
        st.sidebar.success(f"✅ Found production model: {model_name} v{latest_production.version}")
        
        # Load the model with robust error handling
        model = load_model_robustly(model_uri, client, latest_production)
        
        if model is not None:
            st.sidebar.success("🎯 Model loaded successfully!")
            return model
        else:
            st.sidebar.error("❌ Failed to load model artifacts")
            return None
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model from registry: {e}")
        return None

def load_model_robustly(model_uri, client, model_version):
    """Load MLflow model with multiple fallback strategies."""
    strategies = [
        try_direct_loading,
        try_reconstruct_from_run
    ]
    
    last_error = None
    for strategy in strategies:
        try:
            model = strategy(model_uri, client, model_version)
            if model is not None:
                return model
        except Exception as e:
            last_error = e
            continue
    
    # If all strategies fail, show detailed error
    st.sidebar.error(f"❌ All loading strategies failed: {last_error}")
    return None

def try_direct_loading(model_uri, client, model_version):
    """Strategy 1: Try direct loading."""
    try:
        st.sidebar.info("🔄 Attempting direct model loading...")
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Direct loading failed: {e}")
        return None

def try_reconstruct_from_run(model_uri, client, model_version):
    """Strategy 2: Reconstruct model from run information."""
    try:
        st.sidebar.info("🔄 Reconstructing model from run...")
        
        run_id = model_version.run_id
        if not run_id:
            raise Exception("No run ID available for model version")
        
        # Get the run to find artifact location
        run = client.get_run(run_id)
        st.sidebar.info(f"📁 Run ID: {run_id}")
        st.sidebar.info(f"🔬 Experiment ID: {run.info.experiment_id}")
        
        # Try to reconstruct the artifact path - use relative paths
        artifact_base = "./mlruns"
        possible_paths = [
            os.path.join(artifact_base, str(run.info.experiment_id), run_id, "artifacts"),
            os.path.join(artifact_base, str(run.info.experiment_id), run_id, "artifacts", "model"),
        ]
        
        # Add model-specific paths
        model_type = run.data.tags.get("model_type", "unknown")
        st.sidebar.info(f"🤖 Model type: {model_type}")
        
        if model_type == "prophet":
            possible_paths.extend([
                os.path.join(artifact_base, str(run.info.experiment_id), run_id, "artifacts", "prophet_model"),
            ])
        elif model_type == "arima":
            possible_paths.extend([
                os.path.join(artifact_base, str(run.info.experiment_id), run_id, "artifacts", "arima_model"),
            ])
        elif model_type == "lightgbm":
            possible_paths.extend([
                os.path.join(artifact_base, str(run.info.experiment_id), run_id, "artifacts", "lightgbm_model"),
            ])
        
        # Try each possible path
        for path in possible_paths:
            st.sidebar.info(f"🔍 Checking path: {path}")
            if os.path.exists(path):
                st.sidebar.success(f"✅ Found model at: {path}")
                try:
                    return mlflow.pyfunc.load_model(path)
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Failed to load from {path}: {e}")
                    continue
        
        raise Exception(f"❌ Could not find model artifacts in any expected location")
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Reconstruction strategy failed: {e}")
        return None

# In model_registry.py, update the get_model_type_from_registry function:

def get_model_type_from_registry(model_name="BestForecastModels", stage="Production"):
    """Determine the type of model in the registry."""
    try:
        client = MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        staged_models = [mv for mv in model_versions if mv.current_stage == stage]
        
        if staged_models:
            latest_model = max(staged_models, key=lambda x: int(x.version))
            run_id = latest_model.run_id
            
            # Get the run details to check model type
            run = client.get_run(run_id)
            model_type = run.data.tags.get("model_type", "unknown")
            
            # Debug information
            st.sidebar.info(f"🔍 Detected model type: {model_type}")
            st.sidebar.info(f"📝 Run tags: {run.data.tags}")
            
            return model_type
        return "unknown"
    except Exception as e:
        st.sidebar.error(f"❌ Error determining model type: {e}")
        return "unknown"

def fix_mlflow_paths():
    """Utility function to fix MLflow paths after folder moves."""
    try:
        client = MlflowClient()
        
        # Get all model versions
        model_versions = client.search_model_versions("")
        
        fixed_count = 0
        broken_count = 0
        
        st.sidebar.info(f"🔧 Checking {len(model_versions)} model versions...")
        
        for mv in model_versions:
            try:
                # Try to load each model to check if paths are broken
                model_uri = f"models:/{mv.name}/{mv.version}"
                mlflow.pyfunc.load_model(model_uri)
                st.sidebar.success(f"✅ Model {mv.name} v{mv.version} loads successfully")
                fixed_count += 1
            except Exception as e:
                st.sidebar.warning(f"❌ Model {mv.name} v{mv.version} has broken paths: {e}")
                broken_count += 1
        
        if broken_count > 0:
            st.sidebar.error(f"❌ Found {broken_count} models with broken paths out of {len(model_versions)} total models.")
            st.sidebar.info("💡 You need to retrain the models with broken paths.")
        else:
            st.sidebar.success(f"🎉 All {len(model_versions)} model paths are valid!")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error checking MLflow paths: {e}")

def recreate_model_registry():
    """Completely recreate the model registry by re-registering existing runs."""
    try:
        client = MlflowClient()
        
        # Get all runs from the best_models experiment
        experiment = client.get_experiment_by_name("best_models")
        if not experiment:
            st.sidebar.error("❌ best_models experiment not found")
            return
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        st.sidebar.info(f"🔍 Found {len(runs)} runs in best_models experiment")
        
        for run in runs:
            try:
                model_type = run.data.tags.get("model_type", "unknown")
                run_id = run.info.run_id
                
                st.sidebar.info(f"🔄 Re-registering {model_type} model from run {run_id}")
                
                # Construct artifact path
                artifact_path = f"./mlruns/{experiment.experiment_id}/{run_id}/artifacts"
                
                if model_type == "prophet":
                    model_artifact_path = f"{artifact_path}/prophet_model"
                    registered_name = "BestForecastModels"
                elif model_type == "arima":
                    model_artifact_path = f"{artifact_path}/arima_model"
                    registered_name = "BestForecastModels"
                elif model_type == "lightgbm":
                    model_artifact_path = f"{artifact_path}/lightgbm_model"
                    registered_name = "BestForecastModels"
                else:
                    st.sidebar.warning(f"⚠️ Unknown model type: {model_type}")
                    continue
                
                # Check if artifact path exists
                if not os.path.exists(model_artifact_path):
                    st.sidebar.warning(f"⚠️ Artifact path not found: {model_artifact_path}")
                    continue
                
                # Register the model
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/artifacts/{model_type}_model",
                    name=registered_name
                )
                
                st.sidebar.success(f"✅ Successfully re-registered {model_type} model")
                
            except Exception as e:
                st.sidebar.error(f"❌ Failed to re-register run {run.info.run_id}: {e}")
                
    except Exception as e:
        st.sidebar.error(f"❌ Error recreating model registry: {e}")

def check_registry_status():
    """Check and display current registry status"""
    try:
        client = MlflowClient()
        
        st.sidebar.subheader("📊 Registry Status")
        
        # Check experiments
        experiments = client.list_experiments()
        st.sidebar.info(f"🔬 Experiments: {len(experiments)}")
        
        # Check registered models
        registered_models = client.search_registered_models()
        st.sidebar.info(f"📝 Registered Models: {len(registered_models)}")
        
        for rm in registered_models:
            st.sidebar.write(f"  - {rm.name}: {len(rm.latest_versions)} versions")
            for mv in rm.latest_versions:
                st.sidebar.write(f"    - v{mv.version}: {mv.current_stage}")
                
    except Exception as e:
        st.sidebar.error(f"❌ Error checking registry status: {e}")

