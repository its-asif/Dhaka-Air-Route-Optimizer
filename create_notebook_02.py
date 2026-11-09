import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Spatiotemporal Air Quality Forecasting & Health-Aware Routing\n",
                "## Step 2: Feature Engineering & Preprocessing\n",
                "**Author:** PhD Research Framework\n",
                "\n",
                "This notebook scientifically constructs the feature space. Based on our EDA and PACF findings, we will generate temporal encodings, autoregressive lag features (t-1, t-2, t-24), rolling statistics, and scale the data for ML and DL compatibility."
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
                "from sklearn.preprocessing import RobustScaler\n",
                "import joblib\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "print(\"Libraries loaded successfully.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Load Cleaned Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "DATA_PATH = '/kaggle/input/aqi-9-cities/AQI_9_Cities_renamed.csv'\n",
                "if not os.path.exists(DATA_PATH):\n",
                "    DATA_PATH = '../AQI_9_Cities_renamed.csv'\n",
                "    \n",
                "df = pd.read_csv(DATA_PATH)\n",
                "df['datetime'] = pd.to_datetime(df['datetime'])\n",
                "df = df.sort_values(by=['city_name', 'datetime']).reset_index(drop=True)\n",
                "print(f\"Dataset Shape: {df.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Cyclical Temporal Encoding\n",
                "Neural networks struggle with ordinal time variables (Hour 23 is next to Hour 0, but numerically far). Sine and Cosine transformations preserve this cyclicality mathematically."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def add_cyclical_features(data):\n",
                "    df_temp = data.copy()\n",
                "    df_temp['hour'] = df_temp['datetime'].dt.hour\n",
                "    df_temp['month'] = df_temp['datetime'].dt.month\n",
                "    df_temp['dayofweek'] = df_temp['datetime'].dt.dayofweek\n",
                "    \n",
                "    # Hour (24)\n",
                "    df_temp['hour_sin'] = np.sin(df_temp['hour'] * (2. * np.pi / 24))\n",
                "    df_temp['hour_cos'] = np.cos(df_temp['hour'] * (2. * np.pi / 24))\n",
                "    \n",
                "    # Month (12)\n",
                "    df_temp['month_sin'] = np.sin((df_temp['month'] - 1) * (2. * np.pi / 12))\n",
                "    df_temp['month_cos'] = np.cos((df_temp['month'] - 1) * (2. * np.pi / 12))\n",
                "    \n",
                "    # Day of Week (7)\n",
                "    df_temp['day_sin'] = np.sin(df_temp['dayofweek'] * (2. * np.pi / 7))\n",
                "    df_temp['day_cos'] = np.cos(df_temp['dayofweek'] * (2. * np.pi / 7))\n",
                "    \n",
                "    # Drop original ordinal columns\n",
                "    df_temp.drop(['hour', 'month', 'dayofweek'], axis=1, inplace=True)\n",
                "    return df_temp\n",
                "\n",
                "df_encoded = add_cyclical_features(df)\n",
                "print(\"Cyclical temporal features generated.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Autoregressive Lags & Rolling Statistics\n",
                "We group by `city_name` to prevent data leakage between cities when shifting rows."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def create_lags_and_rolling(data):\n",
                "    df_out = pd.DataFrame()\n",
                "    target_cols = ['pm10', 'pm2_5', 'aqi']\n",
                "    \n",
                "    for city, city_df in data.groupby('city_name'):\n",
                "        city_df = city_df.sort_values('datetime')\n",
                "        \n",
                "        # Lags (t-1, t-2, t-24) based on PACF results\n",
                "        for col in target_cols:\n",
                "            city_df[f'{col}_lag1'] = city_df[col].shift(1)\n",
                "            city_df[f'{col}_lag2'] = city_df[col].shift(2)\n",
                "            city_df[f'{col}_lag24'] = city_df[col].shift(24)\n",
                "            \n",
                "            # Rolling mean (24h volatility)\n",
                "            city_df[f'{col}_roll24_mean'] = city_df[col].shift(1).rolling(window=24).mean()\n",
                "            city_df[f'{col}_roll24_std'] = city_df[col].shift(1).rolling(window=24).std()\n",
                "            \n",
                "        df_out = pd.concat([df_out, city_df])\n",
                "        \n",
                "    # Drop NaN rows created by shifting/rolling\n",
                "    df_out.dropna(inplace=True)\n",
                "    return df_out.reset_index(drop=True)\n",
                "\n",
                "df_features = create_lags_and_rolling(df_encoded)\n",
                "print(f\"Feature space expanded. New shape: {df_features.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Data Scaling & Feature Selection\n",
                "Using RobustScaler to mitigate the effect of extreme AQI outlier events. Saving the scaler for the production API pipeline."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "drop_cols = ['city_id', 'city_name', 'datetime', 'lat', 'lon', 'aqi']\n",
                "features = [c for c in df_features.columns if c not in drop_cols]\n",
                "target = 'aqi'\n",
                "\n",
                "scaler = RobustScaler()\n",
                "df_features[features] = scaler.fit_transform(df_features[features])\n",
                "\n",
                "# Save artifact for production\n",
                "os.makedirs('/kaggle/working/models', exist_ok=True)\n",
                "joblib.dump(scaler, '/kaggle/working/models/robust_scaler.pkl')\n",
                "\n",
                "# Save engineered dataset\n",
                "df_features.to_csv('/kaggle/working/engineered_dataset.csv', index=False)\n",
                "print(\"Feature engineering complete. Artifacts saved.\")"
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

with open("kaggle_notebooks/02_feature_engineering.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Kaggle Notebook 02 Generated Successfully!")
