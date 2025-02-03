# data_loader.py
import pandas as pd
import streamlit as st
from config import DATASET_PATH

def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        required_columns = {'Make', 'Model', 'Variant', 'Age', 'Distance_numeric', 'Price_numeric'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            st.error(f"The dataset is missing required columns: {missing}")
            st.stop()
        return df
    except FileNotFoundError:
        st.error(f"Dataset file not found at path: {DATASET_PATH}")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("The dataset file is empty.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Error parsing the dataset file. Please check its format.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the dataset: {e}")
        st.stop()
