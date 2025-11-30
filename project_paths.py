import os

def get_model_paths():
    """Get all file paths for the project - Streamlit Cloud compatible"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    return {
        'data_zip': os.path.join(base_dir, 'Data.zip'),
        'actual_dataset': os.path.join(base_dir, 'actual_dataset.csv'),
        'mlflow_db': 'sqlite:///mlflow.db',
        'artifact_root': 'file:./mlruns'
    }