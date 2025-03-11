import pandas as pd
import io
from typing import Tuple, Dict, Any


def load_data(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Load data from an uploaded file (CSV or Excel).
    
    Args:
        uploaded_file: The uploaded file object from Streamlit
        
    Returns:
        Tuple containing:
        - The loaded DataFrame
        - A message with file details
    """
    # Get file details
    file_details = {
        "filename": uploaded_file.name,
        "filetype": uploaded_file.type,
        "filesize": f"{uploaded_file.size / 1024:.2f} KB"
    }
    
    # Determine file type and load accordingly
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.type}")
    
    # Create file details message
    file_details_str = f"{file_details['filename']} ({file_details['filesize']})"
    
    return df, file_details_str


def validate_data(df: pd.DataFrame, label_col: str, desc_col: str) -> Tuple[bool, str]:
    """
    Validate the selected columns in the DataFrame.
    
    Args:
        df: The DataFrame to validate
        label_col: The name of the label column
        desc_col: The name of the description column
        
    Returns:
        Tuple containing:
        - Boolean indicating whether validation passed
        - A message with validation results
    """
    # Check if dataframe is empty
    if df.empty:
        return False, "The uploaded file contains no data."
    
    # Check if selected columns exist
    if label_col not in df.columns or desc_col not in df.columns:
        return False, "The selected columns don't exist in the uploaded file."
    
    # Check for missing values in label column
    if df[label_col].isna().any():
        missing_count = df[label_col].isna().sum()
        return False, f"The label column has {missing_count} missing values. Please fix these before proceeding."
    
    # Check for missing values in description column
    if df[desc_col].isna().any():
        missing_count = df[desc_col].isna().sum()
        percentage = (missing_count / len(df)) * 100
        
        if percentage > 20:
            return False, f"The description column has {missing_count} ({percentage:.1f}%) missing values. Too many missing values."
        else:
            return True, f"Warning: The description column has {missing_count} ({percentage:.1f}%) missing values, but validation passed."
    
    # Ensure description column contains text
    if not all(isinstance(val, str) for val in df[desc_col].dropna()):
        return False, "The description column must contain text values."
    
    # Check for empty strings in description column
    empty_strings = (df[desc_col] == "").sum()
    if empty_strings > 0:
        percentage = (empty_strings / len(df)) * 100
        if percentage > 10:
            return False, f"The description column has {empty_strings} ({percentage:.1f}%) empty strings. Too many empty descriptions."
        else:
            return True, f"Warning: The description column has {empty_strings} ({percentage:.1f}%) empty strings, but validation passed."
    
    # Success message with basic stats
    return True, f"Data validation passed! {len(df)} records with {df[label_col].nunique()} unique labels." 