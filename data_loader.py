import streamlit as st
import pandas as pd
import os
from datetime import date
from project_paths import get_model_paths

@st.cache_data
def load_data():
    """
    Loads data from Data.zip. 
    Returns the dataframe and key date metrics.
    """
    # GET DYNAMIC PATH
    paths = get_model_paths()
    DATA_PATH = paths['data_zip']
    
    if not os.path.exists(DATA_PATH):
        st.warning(f"Data file not found at: {DATA_PATH}")
        return pd.DataFrame(), date.today(), date.today(), pd.Series(dtype='float64'), pd.DataFrame()

    try:
        # Handle zip file
        train = pd.read_csv(DATA_PATH)
        
        # Date conversion
        train["date"] = pd.to_datetime(train["date"], errors="coerce")
        train = train.dropna(subset=['date'])
        train = train.set_index("date")
        train.index = pd.to_datetime(train.index)

        # Basic Metrics
        min_date = train.index.min().date()
        max_date = train.index.max().date()
        
        # State Sorting for Dashboard
        if 'state' in train.columns and 'sales' in train.columns:
            sort_state = train.groupby('state')['sales'].sum().sort_values(ascending=False)
        else:
            sort_state = pd.Series(dtype='float64')

        # Prepare Prophet Dataframe (aggregated by date) for visualization
        if 'sales' in train.columns:
            prophet_df = train.groupby(train.index)['sales'].sum().reset_index()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        else:
            prophet_df = pd.DataFrame(columns=['ds', 'y'])

        return train, min_date, max_date, sort_state, prophet_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), date.today(), date.today(), pd.Series(dtype='float64'), pd.DataFrame()