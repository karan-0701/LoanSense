import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class QNetwork(nn.Module):
    """Q-network for offline RL agent."""
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Actions: Deny=0, Approve=1
        )

    def forward(self, x):
        return self.model(x)


class OfflineRLAgent:
    """Offline RL agent for loan approval."""

    def __init__(self, df, target_col='loan_status', sample_size=100_000, batch_size=256, test_size=0.2, random_state=42):
        self.df = df
        self.target_col = target_col
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state

    def preprocess(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col].values
        self.categorical_cols = [col for col in X.columns if X[col].nunique() <= 20]
        self.numeric_cols = [col for col in X.columns if col not in self.categorical_cols]

        X[self.numeric_cols] = X[self.numeric_cols].fillna(X[self.numeric_cols].median())
        X[self.categorical_cols] = X[self.categorical_cols].fillna("Unknown")

        self.scaler = StandardScaler()
        X_numeric = self.scaler.fit_transform(X[self.numeric_cols])

        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_categorical = self.ohe.fit_transform(X[self.categorical_cols])

        self.X_processed = np.hstack([X_numeric, X_categorical])

        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(y)
        self.y_tensor = torch.tensor(self.y_encoded, dtype=torch.float32)

        self.loan_amt = torch.tensor(X["loan_amnt"].values, dtype=torch.float32)
        self.int_rate = torch.tensor(X["int_rate"].values, dtype=torch.float32)
        print(f"Preprocessing complete. Input shape: {self.X_processed.shape}")

    def train_test_split(self):
        X_train, X_test, y_train, y_test, loan_train, loan_test, rate_train, rate_test = train_test_split(
            self.X_processed, self.y_tensor, self.loan_amt, self.int_rate,
            test_size=self.test_size, random_state=self.random_state
        )
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_train, self.y_test = y_train, y_test
        self.loan_train, self.loan_test = loan_train.reshape(-1, 1), loan_test
        self.rate_train, self.rate_test = rate_train.reshape(-1, 1), rate_test

        self.train_loader = DataLoader(TensorDataset(self.X_train, self.y_train, self.loan_train, self.rate_train),
                                       batch_size=self.batch_size, shuffle=True)
        print(f"Train/Test split done. Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

    def build_model(self):
        self.model = QNetwork(self.X_train.shape[1])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        print("Q-network built.")

    @staticmethod
    def compute_reward(action, loan_status, loan_amt, int_rate):
        reward = torch.zeros_like(action, dtype=torch.float32)
        approve_mask = (action == 1)
        fully_paid_mask = (loan_status == 1)
        defaulted_mask = (loan_status == 0)
        reward[approve_mask & fully_paid_mask] = loan_amt[approve_mask & fully_paid_mask] * int_rate[approve_mask & fully_paid_mask]
        reward[approve_mask & defaulted_mask] = -loan_amt[approve_mask & defaulted_mask]
        return reward

    def train(self, epochs=5):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for xb, yb, loanb, rateb in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                q_values = self.model(xb)
                action = torch.randint(0, 2, (xb.size(0),))
                reward = self.compute_reward(action, yb, loanb.flatten(), rateb.flatten()) / 10000.0
                q_pred = q_values[torch.arange(len(action)), action]
                loss = self.criterion(q_pred, reward)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(self.train_loader):.4f}")

    def evaluate_policy(self):
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(self.X_test)
            actions = torch.argmax(q_values, dim=1)
            rewards = self.compute_reward(actions, self.y_test, self.loan_test, self.rate_test)

            epv = rewards.sum().item()
            estimated_policy_value = rewards.mean().item()
            approve_ratio = (actions == 1).float().mean().item() * 100
            deny_ratio = 100 - approve_ratio

            print(f"Estimated Policy Value: {estimated_policy_value:.4f}")
            print(f"Approve: {approve_ratio:.2f}%, Deny: {deny_ratio:.2f}%")
            return epv, estimated_policy_value

    def save_model(self, path="offline_rl_agent.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"RL agent model saved to {path}")
