# [file name]: data_loader.py
import streamlit as st
import pandas as pd
import zipfile
import os
from project_paths import get_model_paths
import numpy as np

@st.cache_data
def load_data():
    """Load and prepare data for the application"""
    try:
        paths = get_model_paths()
        
        # Try to load from model_dataset.csv first
        if os.path.exists(paths['dataset']):
            train = pd.read_csv(paths['dataset'], parse_dates=['date'])
            st.success("Loaded data from model_dataset.csv")
        else:
            # Fallback: use sample data or file uploader
            st.info("No dataset found. Please upload your dataset CSV file:")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="data_uploader")
            if uploaded_file is not None:
                train = pd.read_csv(uploaded_file, parse_dates=['date'])
                st.success("Dataset loaded successfully from uploaded file!")
            else:
                # Create sample data for demonstration
                st.warning("Using sample data for demonstration. Please upload your own dataset for full functionality.")
                dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
                sample_data = {
                    'date': dates,
                    'sales': 1000 + np.random.normal(0, 100, len(dates)).cumsum(),
                    'state': ['CA'] * len(dates),
                    'city': ['San Francisco'] * len(dates),
                    'store_nbr': [1] * len(dates)
                }
                train = pd.DataFrame(sample_data)
                st.info("Sample data generated for demonstration")
        
        # Basic data preparation
        if 'date' in train.columns:
            train = train.sort_values('date')
            min_date = train['date'].min()
            max_date = train['date'].max()
        else:
            min_date = max_date = None
            
        # Prepare state data if available
        if 'state' in train.columns:
            sort_state = train.groupby('state')['sales'].sum().sort_values(ascending=False)
        else:
            sort_state = pd.Series()
            
        # Prepare Prophet format data
        if 'sales' in train.columns and 'date' in train.columns:
            prophet_df = train.rename(columns={'date': 'ds', 'sales': 'y'})[['ds', 'y']]
        else:
            prophet_df = pd.DataFrame()
        
        return train, min_date, max_date, sort_state, prophet_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty dataframes to prevent crashes
        return pd.DataFrame(), None, None, pd.Series(), pd.DataFrame()

