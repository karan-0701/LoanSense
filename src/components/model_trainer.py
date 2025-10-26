import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


class ModelTrainer:
    """Deep Learning model trainer for binary classification."""

    def __init__(self, df, target_col='loan_status', batch_size=256, test_size=0.2, random_state=42):
        self.df = df
        self.target_col = target_col
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state

    def preprocess_features(self):
        df = self.df.copy()
        self.categorical_cols = [col for col in df.columns if df[col].nunique() <= 20 and col != self.target_col]
        self.numeric_cols = [col for col in df.columns if col not in self.categorical_cols + [self.target_col]]

        df[self.numeric_cols] = df[self.numeric_cols].fillna(df[self.numeric_cols].median())
        df[self.categorical_cols + [self.target_col]] = df[self.categorical_cols + [self.target_col]].fillna("missing")

        # Scale numeric
        self.scaler = StandardScaler()
        X_numeric = self.scaler.fit_transform(df[self.numeric_cols])

        # One-hot encode categorical
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_categorical = self.ohe.fit_transform(df[self.categorical_cols])

        self.X = np.hstack([X_numeric, X_categorical])
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(df[self.target_col])
        self.num_classes = len(self.le.classes_)

        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.long)

        print(f"Preprocessing complete. Input shape: {self.X.shape}, Num classes: {self.num_classes}")

    def train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_tensor, self.y_tensor, test_size=self.test_size, random_state=self.random_state
        )
        self.train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size)
        self.X_test, self.y_test = X_test, y_test

    def build_model(self):
        input_dim = self.X.shape[1]

        class SimpleNN(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_classes)
                )

            def forward(self, x):
                return self.model(x)

        self.model = SimpleNN(input_dim, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        print("Model built.")

    def train(self, epochs=5):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for xb, yb in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(self.train_loader):.4f}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_test)
            y_pred = torch.argmax(preds, dim=1).cpu().numpy()
            y_true = self.y_test.cpu().numpy()
            y_prob = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

    def save_model(self, path="/Users/karandeepsingh/Desktop/Projects/LoanSense/models/DL/dl_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
