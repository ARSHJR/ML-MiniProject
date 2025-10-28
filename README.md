# 🍔 Fast Food Nutrition ML Mini Project

A comprehensive machine learning mini project built with Streamlit for exploring and analyzing fast food nutritional data through 8 different experiments covering visualization, supervised learning, unsupervised learning, and dimensionality reduction.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Context](#dataset-context)
- [Features & Experiments](#features--experiments)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

---

## 🎯 Project Overview

This interactive Streamlit application provides a hands-on learning platform for machine learning concepts using real-world fast food nutrition data. The project implements 8 core experiments spanning data visualization, regression, classification, ensemble methods, clustering, and dimensionality reduction.

**Key Features:**
- ✅ Interactive web interface with Streamlit
- ✅ 8 comprehensive ML experiments
- ✅ Real-time model training and visualization
- ✅ Adjustable hyperparameters and configurations
- ✅ Dynamic feature selection
- ✅ Performance metrics and evaluation
- ✅ Beautiful visualizations with Plotly and Matplotlib

---

## 📊 Dataset Context

### Origin
The dataset (`fastfood_cleaned.csv`) contains nutritional information from various fast-food restaurant menus, including data on calories, fats, carbohydrates, proteins, sodium, and other nutritional metrics per serving.

### Preprocessing Summary

The dataset has been **pre-processed and cleaned**. The following steps have already been completed:

#### Data Cleaning
1. **Standardized column names** to lowercase with underscores
2. **Converted numeric nutrition columns** (calories, fat, carbs, sugar, protein, sodium, etc.) to numeric datatypes
3. **Parsed serving sizes** into a unified `serving_size_g` column (grams)
4. **Dropped rows** where all nutrient values were zero
5. **Filled missing numeric values** with median values (overall or per restaurant)

#### Feature Engineering
The following engineered features have been added:

- **`macro_calories`** = 4 × protein + 4 × carbs + 9 × fat (calculated calorie estimate from macronutrients)
- **`macro_calorie_gap`** = calories - macro_calories (discrepancy between reported and calculated calories)
- **Macro calorie percentages:**
  - `%_protein_cal` - Percentage of calories from protein
  - `%_carb_cal` - Percentage of calories from carbohydrates
  - `%_fat_cal` - Percentage of calories from fat
- **Nutrient density columns:**
  - `*_per_100g` - Nutritional values normalized per 100 grams
  - `*_per_100kcal` - Nutritional values normalized per 100 kilocalories
- **`energy_density_kcal_per_100g`** - Energy density metric
- **`healthy_rule_v1`** - Binary label (1 = relatively healthy):
  - Criteria: calories ≤ 400, sodium ≤ 600mg, saturated fat ≤ 5g, sugar ≤ 10g
- **`category`** - Derived from item name keywords (burger, salad, sandwich, wrap, beverage, dessert, etc.)

### Dataset Statistics
- **Total Rows:** 1,113 records
- **Total Columns:** 28 columns
- **Data Quality:** No missing values in critical columns after preprocessing

> **Note:** The dataset is ready for direct use. Do not re-clean or re-engineer features.

---

## 🧪 Features & Experiments

The application includes 8 comprehensive experiments accessible via the sidebar:

### 1. 📊 Data Visualisation
**Purpose:** Study and implement basic data visualization methods

**Features:**
- Select X and Y variables
- Choose from 4 plot types: Line Plot, Scatter Plot, Histogram, Box Plot
- View summary statistics (mean, std, min, max) for selected columns
- Interactive Plotly visualizations

**Use Case:** Explore relationships and distributions in the dataset

---

### 2. 📈 Linear Regression
**Purpose:** Prediction and error estimation using simple linear regression

**Features:**
- Single independent variable (X) and dependent variable (Y)
- Regression line visualization with scatter plot
- Regression equation display: y = mx + b
- Performance metrics: R², RMSE, MAE, MSE
- Residual plot for error analysis
- Separate train/test set evaluation

**Use Case:** Predict one nutritional value based on another

---

### 3. 🌳 Decision Tree Classification
**Purpose:** Classification with interpretability through decision trees

**Features:**
- Multi-feature selection
- Automatic categorical target handling
- Adjustable max depth hyperparameter
- Feature importance visualization
- Confusion matrix heatmap
- Classification metrics: Accuracy, Precision, Recall, F1-Score
- Cross-validation with configurable folds
- Detailed classification report

**Use Case:** Classify food items (e.g., healthy vs unhealthy)

---

### 4. 🎯 Support Vector Machine (SVM)
**Purpose:** Powerful classification and regression with kernel methods

**Features:**
- Automatic task detection (classification vs regression)
- Kernel selection: Linear, Polynomial, RBF, Sigmoid
- Adjustable C parameter (regularization)
- Feature scaling (StandardScaler)
- Performance metrics for both classification and regression
- Confusion matrix for classification
- Prediction vs actual plot for regression

**Use Case:** Non-linear decision boundaries and robust predictions

---

### 5. 🌲 Ensemble Learning
**Purpose:** Improved prediction through bagging and boosting

**Features:**
- Method selection:
  - Random Forest (Bagging)
  - Gradient Boosting
  - XGBoost (if available)
- Task type selection: Classification or Regression
- Adjustable number of estimators and max depth
- Feature importance ranking
- Comprehensive performance metrics
- Confusion matrix (classification) or prediction plots (regression)

**Use Case:** Achieve higher accuracy through ensemble methods

---

### 6. 📐 Multivariate Nonlinear Regression
**Purpose:** Multi-input, multi-output regression with nonlinear models

**Features:**
- Multiple independent variables (multiple X)
- Multiple dependent variables (multiple Y)
- Model selection:
  - Polynomial Regression (adjustable degree)
  - Random Forest Regressor
- Overall and per-output performance metrics
- R², RMSE, MAE for each output variable
- Predicted vs actual plots for each output
- Variance explained analysis

**Use Case:** Predict multiple nutritional values simultaneously

---

### 7. 🔍 Clustering (K-Means / DBSCAN)
**Purpose:** Discover natural groupings in data (unsupervised learning)

**Features:**

**K-Means:**
- Adjustable number of clusters
- Elbow method plot (optimal k selection)
- Silhouette score analysis
- Cluster visualization (2D with PCA if needed)
- Cluster statistics and distribution

**DBSCAN:**
- Adjustable epsilon and min_samples parameters
- Automatic cluster discovery
- Noise point detection
- Silhouette score calculation

**Common Features:**
- Feature standardization
- PCA visualization for >2 features
- Cluster count and percentage breakdown

**Use Case:** Group similar food items, identify outliers

---

### 8. 🎨 PCA / SVD (Dimensionality Reduction)
**Purpose:** Reduce feature space while preserving variance

**Features:**
- Method selection: PCA or Truncated SVD
- Adjustable number of components
- Scree plot (variance per component)
- Cumulative variance explained plot
- 95% variance threshold indicator
- Component details table
- 2D scatter plot (first two components)
- 3D scatter plot (first three components, if available)
- Feature loadings matrix (PCA)
- Total variance explained metric

**Use Case:** Visualize high-dimensional data, reduce noise, feature extraction

---

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.8+** installed on your system
- **pip** package manager

### Installation Steps

#### Option 1: Automated Setup (Recommended)

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Linux/Mac (Bash):**
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Create a virtual environment (`.venv`)
2. Activate the virtual environment
3. Upgrade pip
4. Install all dependencies from `requirements.txt`

#### Option 2: Manual Setup

1. **Clone or download the project**

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment:**
   
   **Windows (PowerShell):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   .venv\Scripts\activate.bat
   ```
   
   **Linux/Mac:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project uses the following Python packages:

```
streamlit==1.28.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
matplotlib==3.8.2
plotly==5.18.0
seaborn==0.13.0
xgboost==2.0.3
```

---

## 💻 Usage

### Running the Application

1. **Activate the virtual environment** (if not already activated):
   
   **Windows:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   
   **Linux/Mac:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to the URL displayed (typically `http://localhost:8501`)

### Using the Application

1. **Select an experiment** from the sidebar (1-8)
2. **View dataset preview** by expanding the collapsible section
3. **Configure parameters:**
   - Select features/variables
   - Adjust hyperparameters
   - Set train-test split ratio
   - Set random seed for reproducibility
4. **Click the training/run button** to execute the experiment
5. **Analyze results:**
   - View performance metrics
   - Explore interactive visualizations
   - Examine detailed statistics

### Tips
- Start with **Experiment 1 (Data Visualisation)** to understand the dataset
- Use the **random seed** to ensure reproducible results
- Experiment with different **hyperparameters** to see their effects
- Compare **train vs test** metrics to check for overfitting

---

## 📁 Project Structure

```
MiniProject/
│
├── app.py                    # Main Streamlit application
├── utils.py                  # Utility functions (data loading, metrics, etc.)
├── requirements.txt          # Python dependencies
├── setup.ps1                 # PowerShell setup script (Windows)
├── setup.sh                  # Bash setup script (Linux/Mac)
├── fastfood_cleaned.csv      # Dataset (pre-cleaned)
├── README.md                 # This file
└── .venv/                    # Virtual environment (created during setup)
```

### File Descriptions

- **`app.py`**: Main application file containing all 8 experiments with Streamlit UI
- **`utils.py`**: Helper functions for data loading, preprocessing, metrics display
- **`requirements.txt`**: List of required Python packages
- **`setup.ps1`** / **`setup.sh`**: Automated setup scripts for environment creation
- **`fastfood_cleaned.csv`**: Pre-processed dataset (1,113 rows × 28 columns)
- **`README.md`**: Comprehensive project documentation

---

## 🛠️ Technologies Used

### Core Framework
- **Streamlit** - Interactive web application framework

### Machine Learning
- **scikit-learn** - ML algorithms and tools
- **XGBoost** - Gradient boosting library

### Data Manipulation
- **pandas** - Data analysis and manipulation
- **numpy** - Numerical computing

### Visualization
- **Plotly** - Interactive plots and charts
- **Matplotlib** - Static plots and visualizations
- **Seaborn** - Statistical data visualization

---

## 📚 Learning Outcomes

By working with this project, you will learn:

1. **Data Visualization** - Explore data through various plot types
2. **Regression Analysis** - Predict continuous values
3. **Classification** - Categorize data points
4. **Model Evaluation** - Understand metrics like R², accuracy, precision, recall
5. **Ensemble Methods** - Combine models for better performance
6. **Multi-Output Learning** - Handle multiple prediction targets
7. **Clustering** - Discover patterns without labels
8. **Dimensionality Reduction** - Simplify high-dimensional data

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Extend with more experiments
- Add new visualizations
- Improve the UI/UX
- Optimize model performance
- Add model saving/loading functionality

---

## 📝 Notes

- The dataset is **fixed** and pre-processed - no file uploader is included
- All preprocessing has been completed - use the data as-is
- Models are trained in-memory - no persistence between sessions
- Random seeds ensure reproducibility across runs
- Feature selection is dynamic - experiment with different combinations

---

## 📄 License

This project is created for educational purposes.

---

## 👨‍💻 Author

Created as a Machine Learning Mini Project for academic purposes.

---

## 🎓 Acknowledgments

- Dataset sourced from fast food restaurant nutritional information
- Built with modern Python ML and visualization libraries
- Inspired by practical machine learning education

---

**Enjoy exploring the Fast Food Nutrition ML Mini Project! 🍔📊🤖**
