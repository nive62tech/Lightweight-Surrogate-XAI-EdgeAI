import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
x_val = np.load("data/x_val.npy")
cnn_preds = pd.read_csv("results/cnn_predictions.csv")["cnn_prediction"]

# Flatten input if needed
x_val_flat = x_val.reshape(x_val.shape[0], -1)

# Train surrogate
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(x_val_flat, cnn_preds)

# Save surrogate
joblib.dump(tree, "models/surrogate_tree.pkl")

print("Surrogate model trained and saved.")