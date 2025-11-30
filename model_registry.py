import streamlit as st
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os
import tempfile
import shutil

@st.cache_resource
def load_production_model_from_registry(model_name="BestForecastModels", stage="Production"):
    """Loads the production model from MLflow Model Registry with robust path handling."""
    try:
        client = MlflowClient()
        
        # Get the latest production model version
        model_versions = client.search_model_versions(f"name='{model_name}'")
        production_models = [mv for mv in model_versions if mv.current_stage == stage]
        
        if not production_models:
            st.warning(f"No production model found in registry for {model_name}. Checking for any staged model...")
            # Fallback to any model version
            if model_versions:
                latest_version = max(model_versions, key=lambda x: int(x.version))
                model_uri = f"models:/{model_name}/{latest_version.version}"
                st.info(f"Using latest version {latest_version.version} (stage: {latest_version.current_stage})")
            else:
                st.error(f"No models found in registry for {model_name}")
                return None
        else:
            # Get the latest production model
            latest_production = max(production_models, key=lambda x: int(x.version))
            model_uri = f"models:/{model_name}/{latest_production.version}"
            st.success(f"Loaded production model: {model_name} version {latest_production.version}")
        
        # Load the model with robust error handling
        model = load_model_robustly(model_uri, client, latest_production)
        return model
        
    except Exception as e:
        st.error(f"Error loading model from registry: {e}")
        return None

def load_model_robustly(model_uri, client, model_version):
    """Load MLflow model with multiple fallback strategies."""
    strategies = [
        try_direct_loading,
        try_download_model,
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
    
    # If all strategies fail, raise the last error
    raise last_error if last_error else Exception("All loading strategies failed")

def try_direct_loading(model_uri, client, model_version):
    """Strategy 1: Try direct loading."""
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        st.warning(f"Direct loading failed: {e}")
        return None

def try_download_model(model_uri, client, model_version):
    """Strategy 2: Download model artifacts and load locally."""
    try:
        st.info("Attempting to download model artifacts...")
        
        # Create temporary directory for model download
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the model artifacts
            try:
                # Try to download using MLflow client
                model_path = client.download_model_version(
                    model_version.name, 
                    model_version.version, 
                    temp_dir
                )
                st.success(f"Model downloaded to: {model_path}")
                return mlflow.pyfunc.load_model(model_path)
            except Exception as e:
                st.warning(f"MLflow download failed: {e}")
                
            # Alternative: Try to get the run and download artifacts
            run_id = model_version.run_id
            if run_id:
                try:
                    # Get all artifacts from the run
                    artifacts = client.list_artifacts(run_id)
                    for artifact in artifacts:
                        if 'model' in artifact.path.lower() or 'artifacts' in artifact.path:
                            artifact_path = client.download_artifacts(run_id, artifact.path, temp_dir)
                            st.info(f"Downloaded artifact: {artifact.path}")
                    
                    # Try to load from the downloaded artifacts
                    possible_paths = [
                        os.path.join(temp_dir, "model"),
                        os.path.join(temp_dir, "artifacts"),
                        temp_dir
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            try:
                                return mlflow.pyfunc.load_model(path)
                            except Exception:
                                continue
                except Exception as e:
                    st.warning(f"Artifact download failed: {e}")
        
        return None
    except Exception as e:
        st.warning(f"Download strategy failed: {e}")
        return None

def try_reconstruct_from_run(model_uri, client, model_version):
    """Strategy 3: Reconstruct model from run information."""
    try:
        st.info("Attempting to reconstruct model from run...")
        
        run_id = model_version.run_id
        if not run_id:
            raise Exception("No run ID available for model version")
        
        # Get the run to find artifact location
        run = client.get_run(run_id)
        st.info(f"Run ID: {run_id}")
        st.info(f"Experiment ID: {run.info.experiment_id}")
        
        # Try to reconstruct the artifact path
        artifact_base = "mlruns"
        possible_paths = [
            os.path.join(artifact_base, str(run.info.experiment_id), run_id, "artifacts"),
            os.path.join(artifact_base, str(run.info.experiment_id), run_id, "artifacts", "model"),
            os.path.join(".", artifact_base, str(run.info.experiment_id), run_id, "artifacts"),
        ]
        
        # Add model-specific paths
        model_type = run.data.tags.get("model_type", "unknown")
        if model_type == "arima":
            possible_paths.extend([
                os.path.join(artifact_base, str(run.info.experiment_id), run_id, "artifacts", "arima_model"),
            ])
        elif model_type == "lightgbm":
            possible_paths.extend([
                os.path.join(artifact_base, str(run.info.experiment_id), run_id, "artifacts", "lightgbm_model"),
            ])
        
        # Try each possible path
        for path in possible_paths:
            if os.path.exists(path):
                st.success(f"Found model at: {path}")
                try:
                    return mlflow.pyfunc.load_model(path)
                except Exception as e:
                    st.warning(f"Failed to load from {path}: {e}")
                    continue
        
        raise Exception(f"Could not find model artifacts in any expected location. Run ID: {run_id}")
        
    except Exception as e:
        st.warning(f"Reconstruction strategy failed: {e}")
        return None

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
            
            # Handle prophet model type gracefully
            if model_type == "prophet":
                st.warning("⚠️ Prophet model detected but unavailable. Using LightGBM as fallback.")
                return "lightgbm"
                
            return model_type
        return "unknown"
    except Exception as e:
        st.error(f"Error determining model type: {e}")
        return "unknown"

def fix_mlflow_paths():
    """Utility function to fix MLflow paths after folder moves."""
    try:
        client = MlflowClient()
        
        # Get all model versions
        model_versions = client.search_model_versions("")
        
        fixed_count = 0
        broken_count = 0
        
        for mv in model_versions:
            try:
                # Try to load each model to check if paths are broken
                model_uri = f"models:/{mv.name}/{mv.version}"
                mlflow.pyfunc.load_model(model_uri)
                st.success(f"✓ Model {mv.name} v{mv.version} loads successfully")
            except Exception as e:
                st.warning(f"✗ Model {mv.name} v{mv.version} has broken paths: {e}")
                broken_count += 1
        
        if broken_count > 0:
            st.error(f"Found {broken_count} models with broken paths out of {len(model_versions)} total models.")
            st.info("You need to retrain the models with broken paths.")
        else:
            st.success(f"All {len(model_versions)} model paths are valid!")
            
    except Exception as e:
        st.error(f"Error checking MLflow paths: {e}")

def recreate_model_registry():
    """Completely recreate the model registry by re-registering existing runs."""
    try:
        client = MlflowClient()
        
        # Get all runs from the best_models experiment
        experiment = client.get_experiment_by_name("best_models")
        if not experiment:
            st.error("best_models experiment not found")
            return
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        st.info(f"Found {len(runs)} runs in best_models experiment")
        
        for run in runs:
            try:
                model_type = run.data.tags.get("model_type", "unknown")
                run_id = run.info.run_id
                
                st.info(f"Re-registering {model_type} model from run {run_id}")
                
                # Construct artifact path
                artifact_path = f"mlruns/{experiment.experiment_id}/{run_id}/artifacts"
                
                if model_type == "arima":
                    model_artifact_path = f"{artifact_path}/arima_model"
                    registered_name = "BestForecastModels"
                elif model_type == "lightgbm":
                    model_artifact_path = f"{artifact_path}/lightgbm_model"
                    registered_name = "BestForecastModels"
                else:
                    st.warning(f"Skipping unknown model type: {model_type}")
                    continue
                
                # Check if artifact path exists
                if not os.path.exists(model_artifact_path):
                    st.warning(f"Artifact path not found: {model_artifact_path}")
                    continue
                
                # Register the model
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/artifacts/{model_type}_model",
                    name=registered_name
                )
                
                st.success(f"Successfully re-registered {model_type} model")
                
            except Exception as e:
                st.error(f"Failed to re-register run {run.info.run_id}: {e}")
                
    except Exception as e:
        st.error(f"Error recreating model registry: {e}")
