# [file name]: data_loader.py
import streamlit as st
import pandas as pd
import zipfile
import os
import numpy as np
from project_paths import get_model_paths

@st.cache_data
def load_data():
    """Load and prepare data from Data.zip for Dashboard"""
    try:
        paths = get_model_paths()
        data_zip_path = paths['data_zip']
        
        # Check if Data.zip exists
        if os.path.exists(data_zip_path):
            st.success(f"Found Data.zip at {data_zip_path}")
            
            # Extract and load data from zip
            with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
                # List files in the zip
                file_list = zip_ref.namelist()
                st.info(f"Files in zip: {file_list}")
                
                # Look for Data.csv for dashboard
                data_file = None
                for file in file_list:
                    if file.lower() == 'data.csv' or file.lower().endswith('/data.csv'):
                        data_file = file
                        break
                
                if data_file:
                    # Extract and read the Data.csv file
                    with zip_ref.open(data_file) as f:
                        train = pd.read_csv(f)
                    
                    # Try to parse date column if it exists
                    if 'date' in train.columns:
                        train['date'] = pd.to_datetime(train['date'])
                    
                    st.success(f"✅ Successfully loaded dashboard data from {data_file} in Data.zip")
                    
                else:
                    st.error("No Data.csv file found in Data.zip")
                    st.info("Available files: " + ", ".join(file_list))
                    return pd.DataFrame(), None, None, pd.Series(), pd.DataFrame()
                    
        else:
            st.warning(f"Data.zip not found at {data_zip_path}")
            st.info("Please ensure Data.zip is in your project directory for dashboard functionality")
            
            # Fallback: try to load from uploaded file
            st.info("Alternatively, upload your dataset CSV file:")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="data_uploader")
            if uploaded_file is not None:
                train = pd.read_csv(uploaded_file)
                # Try to parse date column if it exists
                if 'date' in train.columns:
                    train['date'] = pd.to_datetime(train['date'])
                st.success("✅ Dataset loaded successfully from uploaded file!")
            else:
                # Create sample data for demonstration
                st.warning("Using sample data for demonstration. Please upload Data.zip or a CSV file for full functionality.")
                dates = pd.date_range(start='2013-01-01', end='2017-08-15', freq='D')
                sample_data = {
                    'date': dates,
                    'sales': 1000 + np.random.normal(0, 100, len(dates)).cumsum(),
                    'state': ['State_A'] * len(dates),
                    'city': ['City_A'] * len(dates),
                    'store_nbr': [1] * len(dates),
                    'family': ['PRODUCT_FAMILY'] * len(dates),
                    'onpromotion': [0] * len(dates)
                }
                train = pd.DataFrame(sample_data)
                st.info("Sample data generated for demonstration")
        
        # Basic data preparation
        if 'date' in train.columns:
            train = train.sort_values('date')
            train = train.set_index('date')  # Set date as index for time series
            min_date = train.index.min()
            max_date = train.index.max()
        else:
            min_date = max_date = None
            
        # Prepare state data if available
        if 'state' in train.columns:
            sort_state = train.groupby('state')['sales'].sum().sort_values(ascending=False)
        else:
            sort_state = pd.Series()
            
        # Prepare Prophet format data
        if 'sales' in train.columns:
            # Reset index to get date as column
            prophet_df = train.reset_index()[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
        else:
            prophet_df = pd.DataFrame()
        
        return train, min_date, max_date, sort_state, prophet_df
        
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        # Return empty dataframes to prevent crashes
        return pd.DataFrame(), None, None, pd.Series(), pd.DataFrame()
