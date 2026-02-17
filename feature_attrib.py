import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load + Preprocess Data
# -----------------------------
def load_data():
    train = pd.read_csv("data/train.csv")

    # Fill missing values
    for col in ["neighbourhood_group", "room_type"]:
        train[col] = train[col].fillna(train[col].mode()[0])

    for col in ["minimum_nights", "amenity_score",
                "number_of_reviews", "availability_365"]:
        train[col] = train[col].fillna(train[col].median())

    target = "price_class"
    cat_cols = ["neighbourhood_group", "room_type"]
    num_cols = ["minimum_nights", "amenity_score",
                "number_of_reviews", "availability_365"]

    X = train[cat_cols + num_cols].copy()
    y = train[target].astype(int).values

    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    feature_names = list(X.columns)

    return X.values.astype(np.float32), y, feature_names, scaler


# -----------------------------
# Model (2 hidden layers)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# Train Model
# -----------------------------
def train_model(X, y):
    device = torch.device("cpu")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MLP(X.shape[1]).to(device)

    Xtr = torch.tensor(X_train)
    ytr = torch.tensor(y_train)
    Xva = torch.tensor(X_val)
    yva = torch.tensor(y_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        model.train()
        optimizer.zero_grad()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_acc = (torch.argmax(model(Xtr), dim=1) == ytr).float().mean().item()
            val_acc = (torch.argmax(model(Xva), dim=1) == yva).float().mean().item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return model, feature_names, scaler


# -----------------------------
# Evaluate on Test Set
# -----------------------------
def evaluate_test(model, feature_names, scaler):
    test = pd.read_csv("data/test.csv")

    for col in ["neighbourhood_group", "room_type"]:
        test[col] = test[col].fillna(test[col].mode()[0])

    for col in ["minimum_nights", "amenity_score",
                "number_of_reviews", "availability_365"]:
        test[col] = test[col].fillna(test[col].median())

    target = "price_class"
    cat_cols = ["neighbourhood_group", "room_type"]
    num_cols = ["minimum_nights", "amenity_score",
                "number_of_reviews", "availability_365"]

    X_test = test[cat_cols + num_cols].copy()
    y_test = test[target].astype(int).values

    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)

    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0

    X_test = X_test[feature_names]
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    X_test_tensor = torch.tensor(X_test.values.astype(np.float32))

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test_tensor), dim=1)
        test_acc = (preds.numpy() == y_test).mean()

    return test_acc


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    X, y, feature_names, scaler = load_data()

    model, feature_names, scaler = train_model(X, y)

    test_accuracy = evaluate_test(model, feature_names, scaler)

    print("\nTest Accuracy:", round(test_accuracy, 4))
