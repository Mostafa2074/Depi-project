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
    """Get model related paths"""
    root = get_project_root()
    
    # Create necessary directories
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    
    return {
        'dataset': os.path.join(root, "model_dataset.csv"),  # For training
        'data_zip': os.path.join(root, "Data.zip"),  # For dashboard
        'actual_dataset': os.path.join(root, "actual_dataset.csv"),
        'models_folder': os.path.join(root, "models"),
        'prophet_model': os.path.join(root, "models", "prophet_tuned_model.pkl"),
        'arima_model': os.path.join(root, "models", "arima_model.pkl"),
        'lightgbm_model': os.path.join(root, "models", "lightgbm_model.pkl"),
    }
