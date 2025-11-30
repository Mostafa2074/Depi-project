# [file name]: data_loader.py
import streamlit as st
import pandas as pd
import zipfile
import os
from project_paths import get_model_paths

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
            # Fallback: use file uploader
            st.info("Upload your dataset CSV file:")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                train = pd.read_csv(uploaded_file, parse_dates=['date'])
                st.success("Dataset loaded successfully!")
            else:
                st.warning("Please upload a dataset CSV file")
                return pd.DataFrame(), None, None, None, pd.DataFrame()
        
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
        prophet_df = train.rename(columns={'date': 'ds', 'sales': 'y'})[['ds', 'y']] if 'sales' in train.columns else pd.DataFrame()
        
        return train, min_date, max_date, sort_state, prophet_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), None, None, None, pd.DataFrame()
