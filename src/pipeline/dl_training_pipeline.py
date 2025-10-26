import os
import pandas as pd
from components.data_preprocessing import preprocess
from components.model_trainer import ModelTrainer


class DLTrainingPipeline:
    """Pipeline for Deep Learning classifier."""

    def __init__(self, raw_csv_path, processed_csv_path=None, model_save_path="final_model/dl_model.pth"):
        self.raw_csv_path = raw_csv_path
        self.processed_csv_path = processed_csv_path
        self.model_save_path = model_save_path

    def run(self, epochs=5):
        # Step 1: Preprocess data
        if self.processed_csv_path and os.path.exists(self.processed_csv_path):
            df = pd.read_csv(self.processed_csv_path)
        else:
            df = preprocess(self.raw_csv_path)
            if self.processed_csv_path:
                os.makedirs(os.path.dirname(self.processed_csv_path), exist_ok=True)
                df.to_csv(self.processed_csv_path, index=False)

        # Step 2: Initialize model trainer
        trainer = ModelTrainer(df)
        trainer.preprocess_features()
        trainer.train_test_split()
        trainer.build_model()

        # Step 3: Train
        trainer.train(epochs=epochs)

        # Step 4: Evaluate
        trainer.evaluate()

        # Step 5: Save model
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        trainer.save_model(self.model_save_path)