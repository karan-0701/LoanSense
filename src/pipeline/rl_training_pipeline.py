import os
import pandas as pd
from components.data_preprocessing import preprocess
from components.offline_rl_agent import OfflineRLAgent


class RLTrainingPipeline:
    """Pipeline for Offline RL agent."""

    def __init__(self, raw_csv_path, processed_csv_path, model_save_path="final_model/rl_agent.pth"):
        self.raw_csv_path = raw_csv_path
        self.processed_csv_path = processed_csv_path
        self.model_save_path = model_save_path

    def run(self, epochs=5):
        # Step 1: Preprocess
        if self.processed_csv_path and os.path.exists(self.processed_csv_path):
            df = pd.read_csv(self.processed_csv_path)
        else:
            df = preprocess(self.raw_csv_path)
            if self.processed_csv_path:
                os.makedirs(os.path.dirname(self.processed_csv_path), exist_ok=True)
                df.to_csv(self.processed_csv_path, index=False)

        # Step 2: Initialize agent
        agent = OfflineRLAgent(df)
        agent.preprocess()
        agent.train_test_split()
        agent.build_model()

        # Step 3: Train RL agent
        agent.train(epochs=epochs)

        # Step 4: Evaluate
        agent.evaluate_policy()

        # Step 5: Save model
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        agent.save_model(self.model_save_path)

