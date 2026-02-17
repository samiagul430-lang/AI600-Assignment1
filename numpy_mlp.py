import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------
# Activations + derivatives
# -------------------------
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid(a):
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0.0, z)

def drelu(z):
    return (z > 0.0).astype(np.float64)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)

# -------------------------
# Loss + accuracy
# -------------------------
def cross_entropy(probs, y):
    n = y.shape[0]
    eps = 1e-12
    return -np.mean(np.log(probs[np.arange(n), y] + eps))

def accuracy(probs, y):
    preds = np.argmax(probs, axis=1)
    return (preds == y).mean()

# -------------------------
# MLP (2 hidden layers)
# -------------------------
class MLP2Hidden:
    def __init__(self, input_dim, h1=64, h2=32, num_classes=4, activation="sigmoid", seed=42):
        rng = np.random.default_rng(seed)

        if activation == "relu":
            self.W1 = rng.normal(0, np.sqrt(2 / input_dim), size=(input_dim, h1))
            self.W2 = rng.normal(0, np.sqrt(2 / h1), size=(h1, h2))
        else:
            self.W1 = rng.normal(0, np.sqrt(1 / input_dim), size=(input_dim, h1))
            self.W2 = rng.normal(0, np.sqrt(1 / h1), size=(h1, h2))

        self.W3 = rng.normal(0, np.sqrt(1 / h2), size=(h2, num_classes))

        self.b1 = np.zeros((1, h1))
        self.b2 = np.zeros((1, h2))
        self.b3 = np.zeros((1, num_classes))

        self.activation = activation

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1) if self.activation == "relu" else sigmoid(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2) if self.activation == "relu" else sigmoid(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.P = softmax(self.Z3)
        return self.P

    def backward(self, X, y):
        n = X.shape[0]

        dZ3 = self.P.copy()
        dZ3[np.arange(n), y] -= 1.0
        dZ3 /= n

        dW3 = self.A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * drelu(self.Z2) if self.activation == "relu" else dA2 * dsigmoid(self.A2)

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * drelu(self.Z1) if self.activation == "relu" else dA1 * dsigmoid(self.A1)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def step(self, grads, lr=0.1):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]
        self.W3 -= lr * grads["dW3"]
        self.b3 -= lr * grads["db3"]

# -------------------------
# Data prep
# -------------------------
def load_and_prepare_data():
    train = pd.read_csv("data/train.csv")

    for col in ["neighbourhood_group", "room_type"]:
        train[col] = train[col].fillna(train[col].mode()[0])

    for col in ["minimum_nights", "amenity_score", "number_of_reviews", "availability_365"]:
        train[col] = train[col].fillna(train[col].median())

    target = "price_class"
    cat_cols = ["neighbourhood_group", "room_type"]
    num_cols = ["minimum_nights", "amenity_score", "number_of_reviews", "availability_365"]

    X = train[cat_cols + num_cols].copy()
    y = train[target].astype(int).values

    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    scaler = StandardScaler()
    X_enc[num_cols] = scaler.fit_transform(X_enc[num_cols])

    return X_enc.values.astype(np.float64), y

# -------------------------
# Train loop (tracks gradients)
# -------------------------
def train_model(activation="sigmoid", lr=0.1, iters=200, h1=64, h2=32, seed=42):
    X, y = load_and_prepare_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = MLP2Hidden(
        input_dim=X_train.shape[1],
        h1=h1, h2=h2,
        num_classes=4,
        activation=activation,
        seed=seed
    )

    train_acc_hist, val_acc_hist = [], []
    gW1_hist, gW2_hist = [], []

    for i in range(iters):
        p_train = model.forward(X_train)
        acc_train = accuracy(p_train, y_train)

        grads = model.backward(X_train, y_train)

        gW1 = float(np.mean(np.abs(grads["dW1"])))
        gW2 = float(np.mean(np.abs(grads["dW2"])))
        gW1_hist.append(gW1)
        gW2_hist.append(gW2)

        model.step(grads, lr=lr)

        p_val = model.forward(X_val)
        acc_val = accuracy(p_val, y_val)

        train_acc_hist.append(acc_train)
        val_acc_hist.append(acc_val)

        if (i + 1) % 50 == 0:
            print(f"[{activation}] iter {i+1}/{iters} | train acc={acc_train:.4f} val acc={acc_val:.4f} "
                  f"| gW1={gW1:.6f} gW2={gW2:.6f}")

    return train_acc_hist, val_acc_hist, gW1_hist, gW2_hist

def main():
    import os
    import matplotlib.pyplot as plt

    # ALWAYS create folder first
    project_root = os.getcwd()
    fig_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print("Saving figures in:", fig_dir)

    # Train both
    sig_train_acc, sig_val_acc, sig_gW1, sig_gW2 = train_model("sigmoid", lr=0.1, iters=200)
    relu_train_acc, relu_val_acc, relu_gW1, relu_gW2 = train_model("relu", lr=0.05, iters=200)

    # Plot 1: Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(sig_train_acc, label="Sigmoid Train Acc")
    plt.plot(sig_val_acc, label="Sigmoid Val Acc")
    plt.plot(relu_train_acc, label="ReLU Train Acc")
    plt.plot(relu_val_acc, label="ReLU Val Acc")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy vs Iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_iters.png", dpi=300)
    print("Saved accuracy_vs_iters.png")
    plt.show()


    # Plot 2: Gradient Magnitudes
    plt.figure(figsize=(8, 5))
    plt.plot(sig_gW1, label="Sigmoid |grad W1|")
    plt.plot(sig_gW2, label="Sigmoid |grad W2|")
    plt.plot(relu_gW1, label="ReLU |grad W1|")
    plt.plot(relu_gW2, label="ReLU |grad W2|")
    plt.xlabel("Iteration")
    plt.ylabel("Mean absolute gradient")
    plt.title("Gradient Magnitude Across Layers vs Iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grad_magnitude.png", dpi=300)
    print("Saved grad_magnitude.png")
    plt.show()


    print("\nFinal Accuracies:")
    print(f"Sigmoid: train={sig_train_acc[-1]:.4f}, val={sig_val_acc[-1]:.4f}")
    print(f"ReLU   : train={relu_train_acc[-1]:.4f}, val={relu_val_acc[-1]:.4f}")

    print("\nFinal Gradient Magnitudes:")
    print(f"Sigmoid: |gW1|={sig_gW1[-1]:.6f}, |gW2|={sig_gW2[-1]:.6f}")
    print(f"ReLU   : |gW1|={relu_gW1[-1]:.6f}, |gW2|={relu_gW2[-1]:.6f}")

if __name__ == "__main__":
    main()
