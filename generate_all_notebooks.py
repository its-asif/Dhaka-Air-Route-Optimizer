"""
Script to sequentially generate all 6 research Kaggle/Jupyter notebooks:
  01. Dataset Analysis & Exploratory Data Analysis (EDA)
  02. Feature Engineering & Temporal Transformations
  03. Machine Learning Baselines & Benchmarking
  04. Deep Learning & Spatiotemporal Forecasting (LSTM, GRU, TCN)
  05. Explainable AI (SHAP & Feature Attribution)
  06. Health-Aware Route Optimization Integration
"""
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

NOTEBOOK_SCRIPTS = [
    "create_notebook_01.py",
    "create_notebook_02.py",
    "create_notebook_03.py",
    "create_notebook_04.py",
    "create_notebook_05.py",
    "create_notebook_06.py",
]

def main():
    print("=" * 60)
    print("Generating all 6 Kaggle Research Notebooks...")
    print("=" * 60)

    for script_name in NOTEBOOK_SCRIPTS:
        script_path = BASE_DIR / script_name
        if not script_path.exists():
            print(f"Error: {script_name} not found at {script_path}")
            sys.exit(1)

        print(f"\n--> Running {script_name}...")
        result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
        if result.returncode != 0:
            print(f"Failed to generate notebook with {script_name}")
            sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("All 6 Kaggle research notebooks generated successfully in 'kaggle_notebooks/'!")
    print("=" * 60)

if __name__ == "__main__":
    main()
