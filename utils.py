"""
Utility functions for the Fast Food Nutrition ML Mini Project
Provides data loading, preprocessing, and visualization helpers.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path


@st.cache_data
def load_data():
    """
    Load the fastfood_cleaned.csv dataset.
    Uses caching to avoid reloading on every interaction.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    data_path = Path(__file__).parent / "fastfood_cleaned.csv"
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at {data_path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()


def get_numeric_columns(df):
    """
    Get list of numeric column names from dataframe.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df):
    """
    Get list of categorical/object column names from dataframe.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        list: List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def show_dataset_preview(df):
    """
    Display a collapsible dataset preview with summary statistics.
    
    Args:
        df: pandas DataFrame to preview
    """
    with st.expander("📊 Dataset Preview & Summary Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.write("**First 10 rows:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.write("**Summary Statistics (Numeric Columns):**")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.write("**Column Information:**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values.astype(str),
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Unique': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)


def get_train_test_split_params():
    """
    Display UI controls for train-test split parameters.
    
    Returns:
        tuple: (test_size, random_state)
    """
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider(
            "Test Size Ratio",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of dataset to use for testing (e.g., 0.2 = 20%)"
        )
    with col2:
        random_state = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            help="Random seed for reproducible train-test splits"
        )
    return test_size, int(random_state)


def is_classification_task(y):
    """
    Determine if the target variable suggests a classification task.
    
    Args:
        y: Target variable (pandas Series or array)
        
    Returns:
        bool: True if classification, False if regression
    """
    if isinstance(y, pd.Series):
        y_data = y
    else:
        y_data = pd.Series(y)
    
    # Non-numeric is always classification
    if not pd.api.types.is_numeric_dtype(y_data):
        return True
    
    # Check number of unique values
    n_unique = y_data.nunique()
    
    # Classification if <= 20 unique values or boolean
    return n_unique <= 20 or y_data.dtype == bool


def prepare_features_target(df, feature_cols, target_col):
    """
    Prepare features and target, handling missing values.
    
    Args:
        df: DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        
    Returns:
        tuple: (X, y) with cleaned data
    """
    # Select features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing values
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y


def encode_categorical_target(y):
    """
    Encode categorical target variable for classification.
    
    Args:
        y: Target variable
        
    Returns:
        tuple: (encoded_y, label_encoder or None, class_names)
    """
    if pd.api.types.is_numeric_dtype(y):
        return y, None, None
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le, le.classes_


def display_metrics_regression(y_true, y_pred, title="Regression Metrics"):
    """
    Display regression metrics in a formatted way.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Title for the metrics section
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    st.write(f"**{title}**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.4f}")
    with col3:
        st.metric("MAE", f"{mae:.4f}")
    with col4:
        st.metric("MSE", f"{mse:.4f}")
    
    return {"r2": r2, "rmse": rmse, "mae": mae, "mse": mse}


def display_metrics_classification(y_true, y_pred, class_names=None):
    """
    Display classification metrics in a formatted way.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        class_names: Optional list of class names
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle multi-class vs binary
    avg_method = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
    precision = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg_method, zero_division=0)
    
    st.write("**Classification Metrics**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("Precision", f"{precision:.4f}")
    with col3:
        st.metric("Recall", f"{recall:.4f}")
    with col4:
        st.metric("F1-Score", f"{f1:.4f}")
    
    # Classification report
    st.write("**Detailed Classification Report:**")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    st.text(report)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
