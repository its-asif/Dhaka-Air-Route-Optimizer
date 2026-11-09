import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Spatiotemporal Air Quality Forecasting & Health-Aware Routing\n",
                "## Step 3: Machine Learning Baselines & Benchmarking\n",
                "**Author:** PhD Research Framework\n",
                "\n",
                "This notebook establishes the foundational benchmarking for the thesis. We will train and evaluate 5 classic and ensemble Machine Learning models using a strict chronological time-series split to prevent future leakage. The models are evaluated using RMSE, MAE, R², and a Custom Accuracy Formulation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from sklearn.svm import SVR\n",
                "from xgboost import XGBRegressor\n",
                "from lightgbm import LGBMRegressor\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
                "import time\n",
                "import joblib\n",
                "import os\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "plt.style.use('ggplot')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Load Engineered Dataset & Perform Chronological Split\n",
                "For time-series forecasting, random splitting (`train_test_split`) causes data leakage. We must split chronologically."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "DATA_PATH = '/kaggle/input/aqi-engineered-data/engineered_dataset.csv'\n",
                "if not os.path.exists(DATA_PATH):\n",
                "    DATA_PATH = '/kaggle/working/engineered_dataset.csv' # Fallback if run sequentially\n",
                "    if not os.path.exists(DATA_PATH):\n",
                "        DATA_PATH = '../engineered_dataset.csv'\n",
                "\n",
                "try:\n",
                "    df = pd.read_csv(DATA_PATH)\n",
                "    print(f\"Loaded dataset shape: {df.shape}\")\n",
                "except FileNotFoundError:\n",
                "    print(\"Error: engineered_dataset.csv not found. Please run Notebook 02 first.\")\n",
                "\n",
                "# Strict Chronological Split (70% Train, 15% Val, 15% Test)\n",
                "# Note: Data is assumed to be ordered by time per city from Notebook 02.\n",
                "n = len(df)\n",
                "train_idx = int(n * 0.7)\n",
                "val_idx = int(n * 0.85)\n",
                "\n",
                "train_df = df.iloc[:train_idx]\n",
                "val_df = df.iloc[train_idx:val_idx]\n",
                "test_df = df.iloc[val_idx:]\n",
                "\n",
                "features = [c for c in df.columns if c not in ['aqi', 'datetime', 'city_id', 'city_name', 'lat', 'lon']]\n",
                "target = 'aqi'\n",
                "\n",
                "X_train, y_train = train_df[features], train_df[target]\n",
                "X_val, y_val = val_df[features], val_df[target]\n",
                "X_test, y_test = test_df[features], test_df[target]\n",
                "\n",
                "print(f\"Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]} | Test size: {X_test.shape[0]}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Thesis Evaluation Metrics\n",
                "Defining the mathematical formulations for evaluation. For thesis presentation, we define Accuracy as: $100 - MAPE$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluate_model(y_true, y_pred, model_name):\n",
                "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
                "    mae = mean_absolute_error(y_true, y_pred)\n",
                "    r2 = r2_score(y_true, y_pred)\n",
                "    \n",
                "    # Custom Accuracy Formula for Thesis: 100 - MAPE\n",
                "    # Prevent division by zero\n",
                "    non_zero = y_true != 0\n",
                "    if np.any(non_zero):\n",
                "        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100\n",
                "    else:\n",
                "        mape = 0.0\n",
                "    accuracy = max(0, 100 - mape)\n",
                "    \n",
                "    return {\n",
                "        'Model': model_name,\n",
                "        'RMSE': rmse,\n",
                "        'MAE': mae,\n",
                "        'R2 Score': r2,\n",
                "        'Accuracy (%)': accuracy\n",
                "    }"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Model Training & Benchmarking\n",
                "Training the 5 specified Machine Learning models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models = {\n",
                "    'Linear Regression': LinearRegression(),\n",
                "    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),\n",
                "    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),\n",
                "    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1),\n",
                "    'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)\n",
                "}\n",
                "\n",
                "results = []\n",
                "trained_models = {}\n",
                "\n",
                "for name, model in models.items():\n",
                "    print(f\"Training {name}...\")\n",
                "    start_time = time.time()\n",
                "    \n",
                "    # SVR is extremely slow on large datasets, we use a subset if needed, but we'll run full for research rigor\n",
                "    model.fit(X_train, y_train)\n",
                "    \n",
                "    train_time = time.time() - start_time\n",
                "    \n",
                "    # Inference on Test Set\n",
                "    y_pred = model.predict(X_test)\n",
                "    metrics = evaluate_model(y_test.values, y_pred, name)\n",
                "    metrics['Training Time (s)'] = round(train_time, 2)\n",
                "    \n",
                "    results.append(metrics)\n",
                "    trained_models[name] = model\n",
                "    print(f\"{name} Completed. RMSE: {metrics['RMSE']:.4f}\")\n",
                "\n",
                "results_df = pd.DataFrame(results)\n",
                "display(results_df.sort_values('RMSE'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Thesis Visualization: Model Comparison Plots"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "os.makedirs('/kaggle/working/figures', exist_ok=True)\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "# RMSE Comparison\n",
                "sns.barplot(x='RMSE', y='Model', data=results_df.sort_values('RMSE', ascending=False), ax=axes[0], palette='Blues_r')\n",
                "axes[0].set_title('Model Comparison: RMSE (Lower is Better)')\n",
                "axes[0].set_xlabel('Root Mean Squared Error')\n",
                "\n",
                "# R2 Score Comparison\n",
                "sns.barplot(x='R2 Score', y='Model', data=results_df.sort_values('R2 Score'), ax=axes[1], palette='Greens')\n",
                "axes[1].set_title('Model Comparison: R² Score (Higher is Better)')\n",
                "axes[1].set_xlabel('R² Score')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('/kaggle/working/figures/ml_benchmark_comparison.png', dpi=300)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Prediction Error Distribution (Residuals)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "best_model_name = results_df.sort_values('RMSE').iloc[0]['Model']\n",
                "best_model = trained_models[best_model_name]\n",
                "\n",
                "y_pred_best = best_model.predict(X_test)\n",
                "residuals = y_test.values - y_pred_best\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.histplot(residuals, bins=50, kde=True, color='purple')\n",
                "plt.title(f'Residual Distribution for Best ML Model: {best_model_name}')\n",
                "plt.xlabel('Prediction Error (Actual - Predicted AQI)')\n",
                "plt.ylabel('Frequency')\n",
                "plt.axvline(x=0, color='black', linestyle='--')\n",
                "plt.savefig(f'/kaggle/working/figures/{best_model_name.replace(\" \", \"_\")}_residuals.png', dpi=300)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Save Artifacts for Ensembling / Dashboard Deployment"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "os.makedirs('/kaggle/working/models', exist_ok=True)\n",
                "results_df.to_csv('/kaggle/working/ml_leaderboard.csv', index=False)\n",
                "joblib.dump(best_model, f'/kaggle/working/models/best_ml_{best_model_name.replace(\" \", \"_\")}.pkl')\n",
                "print(\"ML Benchmarking complete. Models and Leaderboard saved.\")\n",
                "print(\"Proceed to Notebook 04 for Deep Learning and Spatiotemporal models.\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("kaggle_notebooks/03_ml_baselines.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Kaggle Notebook 03 Generated Successfully!")
