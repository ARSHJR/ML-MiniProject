# 🚀 Quick Start Guide

## Get Started in 3 Steps

### Step 1: Setup Environment
Run the setup script for your platform:

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

### Step 2: Run the Application
```bash
streamlit run app.py
```

### Step 3: Explore!
Open your browser to `http://localhost:8501` and start exploring the 8 experiments!

---

## 🧪 Experiments Overview

1. **Data Visualisation** - Scatter, line, histogram, box plots
2. **Linear Regression** - Simple regression with metrics
3. **Decision Tree Classification** - Feature importance & confusion matrix
4. **Support Vector Machine** - Multiple kernels (linear, poly, rbf)
5. **Ensemble Learning** - Random Forest & Gradient Boosting
6. **Multivariate Nonlinear Regression** - Multi-input, multi-output
7. **Clustering** - K-Means & DBSCAN with visualizations
8. **PCA / SVD** - Dimensionality reduction with variance analysis

---

## 📌 Common Commands

### Activate Virtual Environment
**Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Deactivate Virtual Environment
```bash
deactivate
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Check Streamlit Version
```bash
streamlit --version
```

---

## 💡 Tips

- Use **random seed = 42** for reproducible results
- Start with **test size = 0.2** (20% test, 80% train)
- Try different feature combinations to see their impact
- Check both training and test metrics to detect overfitting
- Experiment with hyperparameters to optimize models

---

## 🐛 Troubleshooting

### Issue: "streamlit: command not found"
**Solution:** Make sure virtual environment is activated

### Issue: "No module named 'streamlit'"
**Solution:** Run `pip install -r requirements.txt`

### Issue: "Dataset not found"
**Solution:** Ensure `fastfood_cleaned.csv` is in the project root directory

### Issue: Script execution policy error (Windows)
**Solution:** Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📊 Dataset Info

- **Rows:** 1,113
- **Columns:** 28
- **Source:** Fast food restaurant menus
- **Status:** Pre-cleaned and feature-engineered (ready to use)

---

## 🎯 Recommended Learning Path

1. Start with **Experiment 1** to explore the data
2. Try **Experiment 2** to understand regression basics
3. Move to **Experiment 3** for classification
4. Compare results with **Experiments 4 & 5** (SVM & Ensembles)
5. Experiment with **Experiment 6** for multi-output learning
6. Discover patterns with **Experiment 7** (Clustering)
7. Visualize high dimensions with **Experiment 8** (PCA/SVD)

---

**Happy Learning! 🎓**
