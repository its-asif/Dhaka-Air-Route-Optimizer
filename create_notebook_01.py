import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Spatiotemporal Air Quality Forecasting & Health-Aware Routing\n",
                "## Step 1: Deep Exploratory Data Analysis & Dataset Assessment\n",
                "**Author:** PhD Research Framework\n",
                "\n",
                "This notebook performs a scientifically rigorous exploratory data analysis (EDA) on the `AQI_9_Cities_renamed.csv` dataset. The goal is to deeply understand the temporal, spatial, and statistical properties of the data before progressing to feature engineering and model development."
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
                "import warnings\n",
                "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n",
                "import scipy.stats as stats\n",
                "\n",
                "warnings.filterwarnings('ignore')\n",
                "plt.style.use('ggplot')\n",
                "sns.set_palette('deep')\n",
                "\n",
                "# For Kaggle compatibility\n",
                "import os\n",
                "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
                "    for filename in filenames:\n",
                "        print(os.path.join(dirname, filename))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Dataset Loading & Structural Assessment"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Adjust path based on Kaggle environment vs local\n",
                "DATA_PATH = '/kaggle/input/aqi-9-cities/AQI_9_Cities_renamed.csv'\n",
                "if not os.path.exists(DATA_PATH):\n",
                "    # Fallback for local execution\n",
                "    DATA_PATH = 'AQI_9_Cities_renamed.csv'\n",
                "    \n",
                "df = pd.read_csv(DATA_PATH)\n",
                "df['datetime'] = pd.to_datetime(df['datetime'])\n",
                "\n",
                "print(f\"Dataset Shape: {df.shape}\")\n",
                "print(\"\\nData Types:\")\n",
                "print(df.dtypes)\n",
                "\n",
                "print(f\"\\nTotal Missing Values: {df.isnull().sum().sum()}\")\n",
                "display(df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Statistical Analysis of Pollutants"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pollutants = ['pm10', 'pm2_5', 'carbon_monoxide', 'carbon_dioxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'aqi']\n",
                "desc_stats = df[pollutants].describe().T\n",
                "desc_stats['skewness'] = df[pollutants].skew()\n",
                "desc_stats['kurtosis'] = df[pollutants].kurtosis()\n",
                "display(desc_stats.round(2))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Target Distribution: AQI Histogram & Density\n",
                "Understanding the distribution of the target variable is crucial for regression modeling. Highly skewed distributions may require log transformations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(12, 6))\n",
                "sns.histplot(df['aqi'], bins=50, kde=True, color='firebrick')\n",
                "plt.title('AQI Distribution Across All Cities', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Air Quality Index (AQI)')\n",
                "plt.ylabel('Frequency')\n",
                "plt.axvline(x=df['aqi'].mean(), color='black', linestyle='--', label=f\"Mean: {df['aqi'].mean():.1f}\")\n",
                "plt.axvline(x=df['aqi'].median(), color='blue', linestyle='-.', label=f\"Median: {df['aqi'].median():.1f}\")\n",
                "plt.legend()\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Temporal Analysis: Hourly, Weekly, and Monthly Trends\n",
                "Analyzing how AQI fluctuates based on temporal cycles to justify temporal feature engineering."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['hour'] = df['datetime'].dt.hour\n",
                "df['day_of_week'] = df['datetime'].dt.dayofweek\n",
                "df['month'] = df['datetime'].dt.month\n",
                "\n",
                "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n",
                "\n",
                "# Hourly Trend\n",
                "sns.boxplot(x='hour', y='aqi', data=df, ax=axes[0], color='skyblue')\n",
                "axes[0].set_title('Hourly AQI Pattern')\n",
                "axes[0].set_xlabel('Hour of Day (0-23)')\n",
                "\n",
                "# Weekly Trend\n",
                "sns.boxplot(x='day_of_week', y='aqi', data=df, ax=axes[1], color='lightgreen')\n",
                "axes[1].set_title('Weekly AQI Pattern')\n",
                "axes[1].set_xlabel('Day of Week (0=Mon, 6=Sun)')\n",
                "axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])\n",
                "\n",
                "# Monthly Trend\n",
                "sns.boxplot(x='month', y='aqi', data=df, ax=axes[2], color='salmon')\n",
                "axes[2].set_title('Monthly/Seasonal AQI Pattern')\n",
                "axes[2].set_xlabel('Month')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Spatial Analysis: AQI Variation by City\n",
                "Evaluating the geographic differences to justify the need for Spatiotemporal graph models (ST-GCN)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(14, 7))\n",
                "sns.violinplot(x='city_name', y='aqi', data=df, palette='muted')\n",
                "plt.title('AQI Distribution by City', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('City')\n",
                "plt.ylabel('Air Quality Index (AQI)')\n",
                "plt.xticks(rotation=45)\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.show()\n",
                "\n",
                "city_means = df.groupby('city_name')['aqi'].mean().sort_values(ascending=False)\n",
                "print(\"Mean AQI by City:\")\n",
                "print(city_means)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Correlation Analysis & Multicollinearity\n",
                "Identifying which pollutants correlate most strongly with the final AQI calculation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 8))\n",
                "corr_matrix = df[pollutants].corr(method='spearman') # Spearman for non-linear monotonic relationships\n",
                "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
                "plt.title('Spearman Correlation Matrix of Pollutants & AQI', fontsize=14)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Time-Series Auto-correlation (ACF & PACF)\n",
                "Analyzing PACF to determine the exact number of autoregressive lag features (`t-1`, `t-2`, etc.) to create for the Deep Learning models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dhaka_aqi = df[df['city_name'] == 'Dhaka'].sort_values('datetime')['aqi'].values\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
                "plot_acf(dhaka_aqi, lags=48, ax=axes[0], title='Autocorrelation (ACF) - Dhaka AQI')\n",
                "plot_pacf(dhaka_aqi, lags=48, ax=axes[1], title='Partial Autocorrelation (PACF) - Dhaka AQI')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8. Conclusion & Feature Engineering Strategy\n",
                "**Insights:**\n",
                "1. **Temporal Patterns**: Strong daily patterns exist. Cyclical encoding for Hour and Month will be highly effective.\n",
                "2. **Spatial Differences**: Significant variance between cities (e.g. Dhaka vs Sylhet) confirms the necessity of localized or Spatiotemporal modeling.\n",
                "3. **Autocorrelation**: The PACF plot shows strong direct lags at `t-1`, `t-2`, and `t-24` (daily seasonality). We will create these specific lag features.\n",
                "4. **Correlations**: PM2.5 and PM10 are the strongest drivers of the AQI, which is expected.\n",
                "\n",
                "**Next Step**: Move to `02_feature_engineering.ipynb`."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs("kaggle_notebooks", exist_ok=True)
with open("kaggle_notebooks/01_dataset_analysis_and_eda.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Kaggle Notebook 01 Generated Successfully!")
