# [file name]: project_paths.py
import os
import streamlit as st

def get_project_root():
    """Get the absolute path to the project root directory"""
    return os.path.dirname(os.path.abspath(__file__))

def get_data_path(filename):
    """Get path for data files"""
    return os.path.join(get_project_root(), filename)

def get_mlflow_paths():
    """Get MLflow related paths - using relative paths"""
    root = get_project_root()
    return {
        'db_path': os.path.join(root, "mlflow.db"),
        'mlruns_path': "./mlruns",  # Use relative path
        'tracking_uri': f"sqlite:///{os.path.join(root, 'mlflow.db')}"
    }

def get_model_paths():
    """Get model related paths - using existing mlflow_project structure"""
    root = get_project_root()
    mlflow_project_path = os.path.join(root, "mlflow_project")
    models_folder = os.path.join(mlflow_project_path, "models", "models")
    
    # Ensure directories exist
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    
    return {
        'dataset': os.path.join(root, "model_dataset.csv"),
        'dataset_dvc': os.path.join(root, "model_dataset.csv.dvc"),
        'mlflow_project_path': mlflow_project_path,
        'models_folder': models_folder,
        'prophet_model': os.path.join(models_folder, "prophet_tuned_model.pkl"),
        'arima_model': os.path.join(models_folder, "arima_model.pkl"),
        'lightgbm_model': os.path.join(models_folder, "Light GBM.pkl"),
        'actual_dataset': os.path.join(root, "actual_dataset.csv"),
        'data_zip': os.path.join(root, "Data.zip"),
        'models_folder_dvc': os.path.join(mlflow_project_path, "models", "models.dvc")
    }
