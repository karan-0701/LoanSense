from pipeline.dl_training_pipeline import DLTrainingPipeline
from pipeline.rl_training_pipeline import RLTrainingPipeline
from components.data_preprocessing import preprocess
from components.data_download import download_file
import os
import pandas as pd
import argparse

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

RAW_CSV = os.path.join(PROJECT_ROOT, "data/raw/loan_dataset.csv")
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data/processed/processed_sample.csv")
DL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/deep_learning/dl_model.pth")
RL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/reinforcement_learning/rl_model.pth")

DOWNLOAD_URL = "http://bit.ly/47pFVe2"


def main(model_type: str):
    print("\n" + "="*40)
    print("STEP 0: Download Raw Data")
    print("="*40 + "\n")

    if not os.path.exists(RAW_CSV):
        download_file(DOWNLOAD_URL, RAW_CSV)
    else:
        print(f"Raw CSV already exists: {RAW_CSV}\n")

    print("\n" + "="*40)
    print("STEP 1: Preprocessing Data")
    print("="*40 + "\n")

    if os.path.exists(PROCESSED_CSV):
        df = pd.read_csv(PROCESSED_CSV)
        print(f"Using existing processed file: {PROCESSED_CSV}\n")
    else:
        df = preprocess(RAW_CSV)
        os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
        df.to_csv(PROCESSED_CSV, index=False)
        print(f"Processed data saved to: {PROCESSED_CSV}\n")

    if model_type in ["dl", "both"]:
        print("\n" + "="*40)
        print("STEP 2: Training DL Model")
        print("="*40 + "\n")
        dl_pipeline = DLTrainingPipeline(RAW_CSV, PROCESSED_CSV, DL_MODEL_PATH)
        dl_pipeline.run(epochs=5)

    if model_type in ["rl", "both"]:
        print("\n" + "="*40)
        print("STEP 3: Training RL Agent")
        print("="*40 + "\n")
        rl_pipeline = RLTrainingPipeline(RAW_CSV, PROCESSED_CSV, RL_MODEL_PATH)
        rl_pipeline.run(epochs=10)

    print("\n" + "="*40)
    print("PIPELINE COMPLETED")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dl", "rl", "both"],
        default="both",
        help="Choose which model to train: dl, rl, or both (default: both)"
    )
    args = parser.parse_args()
    main(args.model)
