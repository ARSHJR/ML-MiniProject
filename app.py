"""
Fast Food Nutrition ML Mini Project
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score, silhouette_samples
)
import seaborn as sns

# Import utility functions
from utils import (
    load_data, get_numeric_columns, get_categorical_columns,
    show_dataset_preview, get_train_test_split_params,
    is_classification_task, prepare_features_target,
    encode_categorical_target, display_metrics_regression,
    display_metrics_classification
)

# Page configuration
st.set_page_config(
    page_title="Fast Food Nutrition ML Project",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6347;
        text-align: center;
        margin-bottom: 1rem;
    }
    .experiment-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4682B4;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">🍔 Fast Food Nutrition ML Mini Project</p>', unsafe_allow_html=True)

# Load data
df = load_data()

# Sidebar
st.sidebar.title("🧪 Experiments")
st.sidebar.markdown("Select an experiment to run:")

experiments = [
    "1. Data Visualisation",
    "2. Linear Regression",
    "3. Decision Tree Classification",
    "4. Support Vector Machine",
    "5. Ensemble Learning",
    "6. Multivariate Nonlinear Regression",
    "7. Clustering (K-Means / DBSCAN)",
    "8. PCA / SVD"
]

selected_experiment = st.sidebar.radio("", experiments)

st.sidebar.markdown("---")
st.sidebar.info(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# Experiment implementations
# ============================================================================
# EXPERIMENT 1: DATA VISUALISATION
# ============================================================================
if selected_experiment == experiments[0]:
    st.markdown('<p class="experiment-header">📊 Experiment 1: Data Visualisation</p>', unsafe_allow_html=True)
    st.write("Study and implement basic data visualisation methods.")
    
    show_dataset_preview(df)
    
    st.markdown("---")
    st.subheader("Visualisation Controls")
    
    numeric_cols = get_numeric_columns(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_var = st.selectbox("Select X Variable", numeric_cols, index=0)
    
    with col2:
        y_var = st.selectbox("Select Y Variable", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    
    with col3:
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Scatter Plot", "Line Plot", "Histogram", "Box Plot"]
        )
    
    st.markdown("---")
    
    # Create visualisation
    if plot_type == "Scatter Plot":
        fig = px.scatter(
            df, x=x_var, y=y_var,
            title=f"Scatter Plot: {x_var} vs {y_var}",
            labels={x_var: x_var, y_var: y_var},
            opacity=0.6,
            color_discrete_sequence=['#FF6347']
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    elif plot_type == "Line Plot":
        fig = px.line(
            df.sort_values(x_var), x=x_var, y=y_var,
            title=f"Line Plot: {x_var} vs {y_var}",
            labels={x_var: x_var, y_var: y_var}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    elif plot_type == "Histogram":
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(
                df, x=x_var, nbins=30,
                title=f"Histogram: {x_var}",
                labels={x_var: x_var},
                color_discrete_sequence=['#FF6347']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            fig = px.histogram(
                df, x=y_var, nbins=30,
                title=f"Histogram: {y_var}",
                labels={y_var: y_var},
                color_discrete_sequence=['#4682B4']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Box Plot":
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.box(
                df, y=x_var,
                title=f"Box Plot: {x_var}",
                color_discrete_sequence=['#FF6347']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            fig = px.box(
                df, y=y_var,
                title=f"Box Plot: {y_var}",
                color_discrete_sequence=['#4682B4']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("---")
    st.subheader("Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{x_var} Statistics:**")
        stats_x = df[x_var].describe()
        st.dataframe(stats_x, use_container_width=True)
    
    with col2:
        st.write(f"**{y_var} Statistics:**")
        stats_y = df[y_var].describe()
        st.dataframe(stats_y, use_container_width=True)

# ============================================================================
# EXPERIMENT 2: LINEAR REGRESSION
# ============================================================================
elif selected_experiment == experiments[1]:
    st.markdown('<p class="experiment-header">📈 Experiment 2: Linear Regression</p>', unsafe_allow_html=True)
    st.write("Prediction and error estimation using simple linear regression.")
    
    show_dataset_preview(df)
    
    st.markdown("---")
    st.subheader("Model Configuration")
    
    numeric_cols = get_numeric_columns(df)
    
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Select Independent Variable (X)", numeric_cols, index=0)
    with col2:
        y_var = st.selectbox("Select Dependent Variable (Y)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    
    test_size, random_state = get_train_test_split_params()
    
    if st.button("🚀 Train Linear Regression Model", type="primary"):
        with st.spinner("Training model..."):
            # Prepare data
            X, y = prepare_features_target(df, [x_var], y_var)
            
            if len(X) == 0:
                st.error("No valid data after removing missing values.")
            else:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Display results
                st.success("✅ Model trained successfully!")
                
                st.markdown("---")
                st.subheader("Regression Equation")
                coef = model.coef_[0]
                intercept = model.intercept_
                st.latex(f"y = {coef:.4f} \\times x + {intercept:.4f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Coefficient (Slope)", f"{coef:.4f}")
                with col2:
                    st.metric("Intercept", f"{intercept:.4f}")
                
                st.markdown("---")
                st.subheader("Model Performance")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Training Set:**")
                    display_metrics_regression(y_train, y_pred_train)
                
                with col2:
                    st.write("**Test Set:**")
                    display_metrics_regression(y_test, y_pred_test)
                
                st.markdown("---")
                st.subheader("Visualisations")
                
                # Scatter plot with regression line
                fig = go.Figure()
                
                # Test data points
                fig.add_trace(go.Scatter(
                    x=X_test[x_var], y=y_test,
                    mode='markers',
                    name='Actual (Test)',
                    marker=dict(color='blue', size=8, opacity=0.6)
                ))
                
                # Regression line
                X_range = np.linspace(X[x_var].min(), X[x_var].max(), 100).reshape(-1, 1)
                X_range_df = pd.DataFrame(X_range, columns=[x_var])
                y_range = model.predict(X_range_df)
                
                fig.add_trace(go.Scatter(
                    x=X_range.flatten(), y=y_range,
                    mode='lines',
                    name='Regression Line',
                    line=dict(color='red', width=3)
                ))
                
                fig.update_layout(
                    title=f"Linear Regression: {y_var} vs {x_var}",
                    xaxis_title=x_var,
                    yaxis_title=y_var,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual plot
                residuals = y_test - y_pred_test
                
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=y_pred_test, y=residuals,
                    mode='markers',
                    marker=dict(color='purple', size=8, opacity=0.6)
                ))
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                fig_res.update_layout(
                    title="Residual Plot",
                    xaxis_title="Predicted Values",
                    yaxis_title="Residuals",
                    height=400
                )
                st.plotly_chart(fig_res, use_container_width=True)

# ============================================================================
# EXPERIMENT 3: DECISION TREE CLASSIFICATION
# ============================================================================
elif selected_experiment == experiments[2]:
    st.markdown('<p class="experiment-header">🌳 Experiment 3: Decision Tree Classification</p>', unsafe_allow_html=True)
    st.write("Classification using Decision Trees with feature importance analysis.")
    
    show_dataset_preview(df)
    
    st.markdown("---")
    st.subheader("Model Configuration")
    
    numeric_cols = get_numeric_columns(df)
    all_cols = df.columns.tolist()
    
    feature_cols = st.multiselect(
        "Select Feature Columns",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols[:1]
    )
    
    target_col = st.selectbox("Select Target Column", all_cols)
    
    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider("Max Depth", 1, 20, 5)
    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
    
    test_size, random_state = get_train_test_split_params()
    
    if st.button("🚀 Train Decision Tree Classifier", type="primary"):
        if len(feature_cols) == 0:
            st.error("Please select at least one feature column.")
        else:
            with st.spinner("Training model..."):
                # Prepare data
                X, y = prepare_features_target(df, feature_cols, target_col)
                
                if len(X) == 0:
                    st.error("No valid data after removing missing values.")
                else:
                    # Encode target if categorical
                    y_encoded, label_encoder, class_names = encode_categorical_target(y)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=random_state
                    )
                    
                    # Train model
                    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Performance metrics
                    st.markdown("---")
                    st.subheader("Model Performance")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Training Set:**")
                        display_metrics_classification(y_train, y_pred_train, class_names)
                    
                    with col2:
                        st.write("**Test Set:**")
                        display_metrics_classification(y_test, y_pred_test, class_names)
                    
                    # Cross-validation
                    st.markdown("---")
                    st.subheader("Cross-Validation")
                    cv_scores = cross_val_score(model, X, y_encoded, cv=cv_folds)
                    st.metric("Mean CV Accuracy", f"{cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                    
                    # Feature importances
                    st.markdown("---")
                    st.subheader("Feature Importances")
                    
                    importances = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importances, x='Importance', y='Feature',
                        orientation='h',
                        title="Feature Importances",
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confusion matrix
                    st.markdown("---")
                    st.subheader("Confusion Matrix")
                    
                    cm = confusion_matrix(y_test, y_pred_test)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=class_names if class_names is not None else 'auto',
                               yticklabels=class_names if class_names is not None else 'auto')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix (Test Set)')
                    st.pyplot(fig)

# ============================================================================
# EXPERIMENT 4: SUPPORT VECTOR MACHINE
# ============================================================================
elif selected_experiment == experiments[3]:
    st.markdown('<p class="experiment-header">🎯 Experiment 4: Support Vector Machine</p>', unsafe_allow_html=True)
    st.write("Classification and regression using Support Vector Machines with different kernels.")
    
    show_dataset_preview(df)
    
    st.markdown("---")
    st.subheader("Model Configuration")
    
    numeric_cols = get_numeric_columns(df)
    all_cols = df.columns.tolist()
    
    feature_cols = st.multiselect(
        "Select Feature Columns",
        numeric_cols,
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols[:1]
    )
    
    target_col = st.selectbox("Select Target Column", all_cols)
    
    col1, col2 = st.columns(2)
    with col1:
        kernel = st.selectbox("Select Kernel", ["linear", "poly", "rbf", "sigmoid"])
    with col2:
        c_value = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.1)
    
    test_size, random_state = get_train_test_split_params()
    
    if st.button("🚀 Train SVM Model", type="primary"):
        if len(feature_cols) == 0:
            st.error("Please select at least one feature column.")
        else:
            with st.spinner("Training model..."):
                # Prepare data
                X, y = prepare_features_target(df, feature_cols, target_col)
                
                if len(X) == 0:
                    st.error("No valid data after removing missing values.")
                else:
                    # Determine task type
                    is_classification = is_classification_task(y)
                    
                    # Encode target if classification
                    if is_classification:
                        y_encoded, label_encoder, class_names = encode_categorical_target(y)
                        y_to_use = y_encoded
                    else:
                        y_to_use = y
                        class_names = None
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_to_use, test_size=test_size, random_state=random_state
                    )
                    
                    # Scale features (important for SVM)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train model
                    if is_classification:
                        model = SVC(kernel=kernel, C=c_value, random_state=random_state)
                        task_type = "Classification"
                    else:
                        model = SVR(kernel=kernel, C=c_value)
                        task_type = "Regression"
                    
                    model.fit(X_train_scaled, y_train)
                    
                    # Predictions
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    st.success(f"✅ SVM {task_type} model trained successfully!")
                    
                    st.info(f"**Task Type:** {task_type} | **Kernel:** {kernel} | **C:** {c_value}")
                    
                    # Performance metrics
                    st.markdown("---")
                    st.subheader("Model Performance")
                    
                    col1, col2 = st.columns(2)
                    
                    if is_classification:
                        with col1:
                            st.write("**Training Set:**")
                            display_metrics_classification(y_train, y_pred_train, class_names)
                        
                        with col2:
                            st.write("**Test Set:**")
                            display_metrics_classification(y_test, y_pred_test, class_names)
                        
                        # Confusion matrix
                        st.markdown("---")
                        st.subheader("Confusion Matrix")
                        
                        cm = confusion_matrix(y_test, y_pred_test)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                                   xticklabels=class_names if class_names is not None else 'auto',
                                   yticklabels=class_names if class_names is not None else 'auto')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix (Test Set)')
                        st.pyplot(fig)
                        
                    else:
                        with col1:
                            st.write("**Training Set:**")
                            display_metrics_regression(y_train, y_pred_train)
                        
                        with col2:
                            st.write("**Test Set:**")
                            display_metrics_regression(y_test, y_pred_test)
                        
                        # Prediction plot
                        st.markdown("---")
                        st.subheader("Predictions vs Actual")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=y_test, y=y_pred_test,
                            mode='markers',
                            name='Predictions',
                            marker=dict(color='green', size=8, opacity=0.6)
                        ))
                        
                        # Perfect prediction line
                        min_val = min(y_test.min(), y_pred_test.min())
                        max_val = max(y_test.max(), y_pred_test.max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Predicted vs Actual Values",
                            xaxis_title="Actual",
                            yaxis_title="Predicted",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# EXPERIMENT 5: ENSEMBLE LEARNING
# ============================================================================
elif selected_experiment == experiments[4]:
    st.markdown('<p class="experiment-header">🌲 Experiment 5: Ensemble Learning</p>', unsafe_allow_html=True)
    st.write("Bagging and Boosting methods for improved prediction performance.")
    
    show_dataset_preview(df)
    
    st.markdown("---")
    st.subheader("Model Configuration")
    
    numeric_cols = get_numeric_columns(df)
    all_cols = df.columns.tolist()
    
    feature_cols = st.multiselect(
        "Select Feature Columns",
        numeric_cols,
        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols[:1]
    )
    
    target_col = st.selectbox("Select Target Column", all_cols)
    
    col1, col2 = st.columns(2)
    with col1:
        task_type = st.selectbox("Task Type", ["Classification", "Regression"])
    with col2:
        ensemble_method = st.selectbox(
            "Ensemble Method",
            ["Random Forest (Bagging)", "Gradient Boosting", "XGBoost (Boosting)"]
        )
    
    col3, col4 = st.columns(2)
    with col3:
        n_estimators = st.slider("Number of Estimators", 10, 200, 100, 10)
    with col4:
        max_depth = st.slider("Max Depth", 1, 20, 5)
    
    test_size, random_state = get_train_test_split_params()
    
    if st.button("🚀 Train Ensemble Model", type="primary"):
        if len(feature_cols) == 0:
            st.error("Please select at least one feature column.")
        else:
            with st.spinner("Training model..."):
                # Prepare data
                X, y = prepare_features_target(df, feature_cols, target_col)
                
                if len(X) == 0:
                    st.error("No valid data after removing missing values.")
                else:
                    # Encode target if classification
                    if task_type == "Classification":
                        y_encoded, label_encoder, class_names = encode_categorical_target(y)
                        y_to_use = y_encoded
                    else:
                        y_to_use = y
                        class_names = None
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_to_use, test_size=test_size, random_state=random_state
                    )
                    
                    # Select model
                    if task_type == "Classification":
                        if "Random Forest" in ensemble_method:
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state
                            )
                        elif "Gradient Boosting" in ensemble_method:
                            model = GradientBoostingClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state
                            )
                        else:  # XGBoost
                            try:
                                from xgboost import XGBClassifier
                                model = XGBClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=random_state
                                )
                            except ImportError:
                                st.warning("XGBoost not available. Using Gradient Boosting instead.")
                                model = GradientBoostingClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=random_state
                                )
                    else:  # Regression
                        if "Random Forest" in ensemble_method:
                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state
                            )
                        elif "Gradient Boosting" in ensemble_method:
                            model = GradientBoostingRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state
                            )
                        else:  # XGBoost
                            try:
                                from xgboost import XGBRegressor
                                model = XGBRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=random_state
                                )
                            except ImportError:
                                st.warning("XGBoost not available. Using Gradient Boosting instead.")
                                model = GradientBoostingRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=random_state
                                )
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    st.success(f"✅ {ensemble_method} model trained successfully!")
                    
                    # Performance metrics
                    st.markdown("---")
                    st.subheader("Model Performance")
                    
                    col1, col2 = st.columns(2)
                    
                    if task_type == "Classification":
                        with col1:
                            st.write("**Training Set:**")
                            display_metrics_classification(y_train, y_pred_train, class_names)
                        
                        with col2:
                            st.write("**Test Set:**")
                            display_metrics_classification(y_test, y_pred_test, class_names)
                    else:
                        with col1:
                            st.write("**Training Set:**")
                            display_metrics_regression(y_train, y_pred_train)
                        
                        with col2:
                            st.write("**Test Set:**")
                            display_metrics_regression(y_test, y_pred_test)
                    
                    # Feature importances
                    st.markdown("---")
                    st.subheader("Feature Importances")
                    
                    importances = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importances, x='Importance', y='Feature',
                        orientation='h',
                        title=f"Feature Importances ({ensemble_method})",
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional visualizations
                    if task_type == "Classification":
                        st.markdown("---")
                        st.subheader("Confusion Matrix")
                        
                        cm = confusion_matrix(y_test, y_pred_test)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                                   xticklabels=class_names if class_names is not None else 'auto',
                                   yticklabels=class_names if class_names is not None else 'auto')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix (Test Set)')
                        st.pyplot(fig)

# ============================================================================
# EXPERIMENT 6: MULTIVARIATE NONLINEAR REGRESSION
# ============================================================================
elif selected_experiment == experiments[5]:
    st.markdown('<p class="experiment-header">📐 Experiment 6: Multivariate Nonlinear Regression</p>', unsafe_allow_html=True)
    st.write("Multi-input, multi-output regression with polynomial features.")
    
    show_dataset_preview(df)
    
    st.markdown("---")
    st.subheader("Model Configuration")
    
    numeric_cols = get_numeric_columns(df)
    
    feature_cols = st.multiselect(
        "Select Independent Variables (Multiple X)",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols[:1]
    )
    
    target_cols = st.multiselect(
        "Select Dependent Variables (Multiple Y)",
        numeric_cols,
        default=[numeric_cols[-1]] if len(numeric_cols) >= 1 else []
    )
    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["Polynomial Regression", "Random Forest Regressor"]
        )
    with col2:
        if model_type == "Polynomial Regression":
            poly_degree = st.slider("Polynomial Degree", 1, 5, 2)
        else:
            n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
    
    test_size, random_state = get_train_test_split_params()
    
    if st.button("🚀 Train Multi-Output Regression Model", type="primary"):
        if len(feature_cols) == 0 or len(target_cols) == 0:
            st.error("Please select at least one feature and one target column.")
        else:
            with st.spinner("Training model..."):
                # Prepare data
                valid_idx = df[feature_cols + target_cols].notna().all(axis=1)
                X = df.loc[valid_idx, feature_cols].values
                y = df.loc[valid_idx, target_cols].values
                
                if len(X) == 0:
                    st.error("No valid data after removing missing values.")
                else:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Train model
                    if model_type == "Polynomial Regression":
                        # Create polynomial features
                        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                        X_train_poly = poly.fit_transform(X_train)
                        X_test_poly = poly.transform(X_test)
                        
                        # Train linear regression on polynomial features
                        model = LinearRegression()
                        model.fit(X_train_poly, y_train)
                        
                        # Predictions
                        y_pred_train = model.predict(X_train_poly)
                        y_pred_test = model.predict(X_test_poly)
                        
                        model_name = f"Polynomial Regression (degree={poly_degree})"
                    else:
                        # Random Forest for multi-output
                        from sklearn.multioutput import MultiOutputRegressor
                        base_model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            random_state=random_state
                        )
                        model = MultiOutputRegressor(base_model)
                        model.fit(X_train, y_train)
                        
                        # Predictions
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)
                        
                        model_name = f"Random Forest (n_estimators={n_estimators})"
                    
                    st.success(f"✅ {model_name} model trained successfully!")
                    
                    st.info(f"**Input Features:** {len(feature_cols)} | **Output Variables:** {len(target_cols)}")
                    
                    # Overall metrics
                    st.markdown("---")
                    st.subheader("Overall Model Performance")
                    
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    
                    # Calculate overall metrics
                    r2_train = r2_score(y_train, y_pred_train, multioutput='variance_weighted')
                    r2_test = r2_score(y_test, y_pred_test, multioutput='variance_weighted')
                    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test, multioutput='uniform_average'))
                    mae_test = mean_absolute_error(y_test, y_pred_test, multioutput='uniform_average')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("R² (Train)", f"{r2_train:.4f}")
                    with col2:
                        st.metric("R² (Test)", f"{r2_test:.4f}")
                    with col3:
                        st.metric("RMSE (Test)", f"{rmse_test:.4f}")
                    with col4:
                        st.metric("MAE (Test)", f"{mae_test:.4f}")
                    
                    # Per-output metrics
                    st.markdown("---")
                    st.subheader("Per-Output Performance")
                    
                    for i, target_name in enumerate(target_cols):
                        with st.expander(f"📊 {target_name}", expanded=True):
                            y_true_output = y_test[:, i] if y_test.ndim > 1 else y_test
                            y_pred_output = y_pred_test[:, i] if y_pred_test.ndim > 1 else y_pred_test
                            
                            r2 = r2_score(y_true_output, y_pred_output)
                            rmse = np.sqrt(mean_squared_error(y_true_output, y_pred_output))
                            mae = mean_absolute_error(y_true_output, y_pred_output)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R²", f"{r2:.4f}")
                            with col2:
                                st.metric("RMSE", f"{rmse:.4f}")
                            with col3:
                                st.metric("MAE", f"{mae:.4f}")
                            
                            # Predicted vs Actual plot
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=y_true_output, y=y_pred_output,
                                mode='markers',
                                name='Predictions',
                                marker=dict(color='purple', size=8, opacity=0.6)
                            ))
                            
                            # Perfect prediction line
                            min_val = min(y_true_output.min(), y_pred_output.min())
                            max_val = max(y_true_output.max(), y_pred_output.max())
                            fig.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title=f"Predicted vs Actual: {target_name}",
                                xaxis_title="Actual",
                                yaxis_title="Predicted",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# EXPERIMENT 7: CLUSTERING
# ============================================================================
elif selected_experiment == experiments[6]:
    st.markdown('<p class="experiment-header">🔍 Experiment 7: Clustering (K-Means / DBSCAN)</p>', unsafe_allow_html=True)
    st.write("Unsupervised learning using K-Means and DBSCAN clustering algorithms.")
    
    show_dataset_preview(df)
    
    st.markdown("---")
    st.subheader("Clustering Configuration")
    
    numeric_cols = get_numeric_columns(df)
    
    feature_cols = st.multiselect(
        "Select Features for Clustering",
        numeric_cols,
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols[:1]
    )
    
    algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN"])
    
    if algorithm == "K-Means":
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters (k)", 2, 10, 3)
        with col2:
            random_state = st.number_input("Random Seed", 0, 9999, 42)
    else:  # DBSCAN
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("Epsilon (eps)", 0.1, 10.0, 0.5, 0.1)
        with col2:
            min_samples = st.slider("Min Samples", 2, 20, 5)
    
    if st.button("🚀 Run Clustering", type="primary"):
        if len(feature_cols) == 0:
            st.error("Please select at least one feature column.")
        else:
            with st.spinner("Running clustering algorithm..."):
                # Prepare data
                X = df[feature_cols].dropna()
                
                if len(X) == 0:
                    st.error("No valid data after removing missing values.")
                else:
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Run clustering
                    if algorithm == "K-Means":
                        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                        labels = model.fit_predict(X_scaled)
                        
                        st.success(f"✅ K-Means clustering complete with {n_clusters} clusters!")
                        
                        # Elbow plot
                        st.markdown("---")
                        st.subheader("Elbow Method")
                        
                        inertias = []
                        K_range = range(2, min(11, len(X)))
                        for k in K_range:
                            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                            kmeans.fit(X_scaled)
                            inertias.append(kmeans.inertia_)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(K_range), y=inertias,
                            mode='lines+markers',
                            marker=dict(size=10, color='blue'),
                            line=dict(color='blue', width=2)
                        ))
                        fig.update_layout(
                            title="Elbow Plot",
                            xaxis_title="Number of Clusters (k)",
                            yaxis_title="Inertia (Within-Cluster Sum of Squares)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Silhouette analysis
                        st.markdown("---")
                        st.subheader("Silhouette Analysis")
                        
                        silhouette_scores = []
                        for k in K_range:
                            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                            cluster_labels = kmeans.fit_predict(X_scaled)
                            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                            silhouette_scores.append(silhouette_avg)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(K_range), y=silhouette_scores,
                            mode='lines+markers',
                            marker=dict(size=10, color='green'),
                            line=dict(color='green', width=2)
                        ))
                        fig.update_layout(
                            title="Silhouette Score vs Number of Clusters",
                            xaxis_title="Number of Clusters (k)",
                            yaxis_title="Silhouette Score",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        silhouette_avg = silhouette_score(X_scaled, labels)
                        st.metric("Silhouette Score (Current k)", f"{silhouette_avg:.4f}")
                        
                    else:  # DBSCAN
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = model.fit_predict(X_scaled)
                        
                        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise = list(labels).count(-1)
                        
                        st.success(f"✅ DBSCAN clustering complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Clusters Found", n_clusters_found)
                        with col2:
                            st.metric("Noise Points", n_noise)
                        
                        if n_clusters_found > 1:
                            # Silhouette score (excluding noise points)
                            mask = labels != -1
                            if mask.sum() > 0:
                                silhouette_avg = silhouette_score(X_scaled[mask], labels[mask])
                                st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
                    
                    # Cluster visualization
                    st.markdown("---")
                    st.subheader("Cluster Visualization")
                    
                    # If more than 2 features, use PCA
                    if len(feature_cols) > 2:
                        pca = PCA(n_components=2)
                        X_viz = pca.fit_transform(X_scaled)
                        viz_labels = [f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                                     f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"]
                        st.info("Using PCA to reduce to 2D for visualization")
                    else:
                        X_viz = X_scaled
                        viz_labels = feature_cols
                    
                    # Create scatter plot
                    plot_df = pd.DataFrame(
                        X_viz[:, :2],
                        columns=['Dimension 1', 'Dimension 2']
                    )
                    plot_df['Cluster'] = labels.astype(str)
                    
                    fig = px.scatter(
                        plot_df,
                        x='Dimension 1',
                        y='Dimension 2',
                        color='Cluster',
                        title=f"Cluster Visualization ({algorithm})",
                        labels={'Dimension 1': viz_labels[0] if len(viz_labels) > 0 else 'Dimension 1',
                               'Dimension 2': viz_labels[1] if len(viz_labels) > 1 else 'Dimension 2'},
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig.update_traces(marker=dict(size=8, opacity=0.7))
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster statistics
                    st.markdown("---")
                    st.subheader("Cluster Statistics")
                    
                    cluster_stats = []
                    unique_labels = sorted(set(labels))
                    for label in unique_labels:
                        mask = labels == label
                        cluster_name = f"Noise" if label == -1 else f"Cluster {label}"
                        cluster_stats.append({
                            'Cluster': cluster_name,
                            'Count': mask.sum(),
                            'Percentage': f"{mask.sum() / len(labels) * 100:.2f}%"
                        })
                    
                    stats_df = pd.DataFrame(cluster_stats)
                    st.dataframe(stats_df, use_container_width=True)

# ============================================================================
# EXPERIMENT 8: PCA / SVD
# ============================================================================
elif selected_experiment == experiments[7]:
    st.markdown('<p class="experiment-header">🎨 Experiment 8: PCA / SVD (Dimensionality Reduction)</p>', unsafe_allow_html=True)
    st.write("Principal Component Analysis and Singular Value Decomposition for dimension reduction.")
    
    show_dataset_preview(df)
    
    st.markdown("---")
    st.subheader("Dimension Reduction Configuration")
    
    numeric_cols = get_numeric_columns(df)
    
    feature_cols = st.multiselect(
        "Select Features for Dimension Reduction",
        numeric_cols,
        default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
    )
    
    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox("Method", ["PCA", "SVD (Truncated)"])
    with col2:
        n_components = st.slider(
            "Number of Components",
            2,
            min(10, len(feature_cols)) if len(feature_cols) > 0 else 2,
            min(3, len(feature_cols)) if len(feature_cols) > 0 else 2
        )
    
    if st.button("🚀 Apply Dimension Reduction", type="primary"):
        if len(feature_cols) == 0:
            st.error("Please select at least one feature column.")
        elif len(feature_cols) < n_components:
            st.error(f"Number of components ({n_components}) cannot exceed number of features ({len(feature_cols)}).")
        else:
            with st.spinner("Applying dimension reduction..."):
                # Prepare data
                X = df[feature_cols].dropna()
                
                if len(X) == 0:
                    st.error("No valid data after removing missing values.")
                else:
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply method
                    if method == "PCA":
                        model = PCA(n_components=n_components)
                        X_transformed = model.fit_transform(X_scaled)
                        variance_ratio = model.explained_variance_ratio_
                        singular_values = model.singular_values_
                    else:  # SVD
                        model = TruncatedSVD(n_components=n_components, random_state=42)
                        X_transformed = model.fit_transform(X_scaled)
                        variance_ratio = model.explained_variance_ratio_
                        singular_values = model.singular_values_
                    
                    st.success(f"✅ {method} completed successfully!")
                    
                    # Variance explained
                    st.markdown("---")
                    st.subheader("Variance Explained")
                    
                    total_variance = variance_ratio.sum()
                    st.metric(
                        f"Total Variance Explained by {n_components} Components",
                        f"{total_variance:.2%}"
                    )
                    
                    # Scree plot
                    st.markdown("---")
                    st.subheader("Scree Plot")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f"PC{i+1}" for i in range(n_components)],
                        y=variance_ratio * 100,
                        marker_color='steelblue',
                        name='Individual'
                    ))
                    fig.update_layout(
                        title="Variance Explained by Each Component",
                        xaxis_title="Principal Component",
                        yaxis_title="Variance Explained (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cumulative variance plot
                    st.markdown("---")
                    st.subheader("Cumulative Variance Explained")
                    
                    cumsum_variance = np.cumsum(variance_ratio) * 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[f"PC{i+1}" for i in range(n_components)],
                        y=cumsum_variance,
                        mode='lines+markers',
                        marker=dict(size=10, color='darkgreen'),
                        line=dict(color='darkgreen', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(0,100,0,0.2)'
                    ))
                    fig.add_hline(y=95, line_dash="dash", line_color="red",
                                 annotation_text="95% threshold")
                    fig.update_layout(
                        title="Cumulative Variance Explained",
                        xaxis_title="Number of Components",
                        yaxis_title="Cumulative Variance Explained (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Component details
                    st.markdown("---")
                    st.subheader("Component Details")
                    
                    component_df = pd.DataFrame({
                        'Component': [f"PC{i+1}" for i in range(n_components)],
                        'Variance Explained (%)': variance_ratio * 100,
                        'Cumulative Variance (%)': cumsum_variance,
                        'Singular Value': singular_values
                    })
                    st.dataframe(component_df, use_container_width=True)
                    
                    # 2D Visualization
                    st.markdown("---")
                    st.subheader("2D Component Visualization")
                    
                    plot_df = pd.DataFrame(
                        X_transformed[:, :2],
                        columns=['PC1', 'PC2']
                    )
                    
                    fig = px.scatter(
                        plot_df,
                        x='PC1',
                        y='PC2',
                        title=f"First Two Principal Components ({method})",
                        labels={
                            'PC1': f"PC1 ({variance_ratio[0]:.1%})",
                            'PC2': f"PC2 ({variance_ratio[1]:.1%})"
                        },
                        opacity=0.6,
                        color_discrete_sequence=['purple']
                    )
                    fig.update_traces(marker=dict(size=6))
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3D Visualization (if at least 3 components)
                    if n_components >= 3:
                        st.markdown("---")
                        st.subheader("3D Component Visualization")
                        
                        plot_df_3d = pd.DataFrame(
                            X_transformed[:, :3],
                            columns=['PC1', 'PC2', 'PC3']
                        )
                        
                        fig = px.scatter_3d(
                            plot_df_3d,
                            x='PC1',
                            y='PC2',
                            z='PC3',
                            title=f"First Three Principal Components ({method})",
                            labels={
                                'PC1': f"PC1 ({variance_ratio[0]:.1%})",
                                'PC2': f"PC2 ({variance_ratio[1]:.1%})",
                                'PC3': f"PC3 ({variance_ratio[2]:.1%})"
                            },
                            opacity=0.7,
                            color_discrete_sequence=['orange']
                        )
                        fig.update_traces(marker=dict(size=3))
                        fig.update_layout(height=700)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature loadings (for PCA)
                    if method == "PCA" and hasattr(model, 'components_'):
                        st.markdown("---")
                        st.subheader("Feature Loadings")
                        
                        loadings_df = pd.DataFrame(
                            model.components_.T,
                            columns=[f"PC{i+1}" for i in range(n_components)],
                            index=feature_cols
                        )
                        
                        st.write("Top feature contributions to each principal component:")
                        st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm', axis=0),
                                   use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🍔 Fast Food Nutrition ML Mini Project | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
