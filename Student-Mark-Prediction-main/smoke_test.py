"""Lightweight repo smoke test used by CI.

This imports the main modules and runs a quick prepare-data call.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    print("Running smoke test: importing modules...")
    import data_processing
    import predict
    import train_models
    import streamlit_app

    print("Loading and preparing data (this reads CSVs)...")
    prepared = data_processing.load_and_prepare_data(ROOT)
    print("Prepared df shape:", prepared['df'].shape)
    print("Smoke test passed.")

if __name__ == '__main__':
    main()
