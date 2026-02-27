import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load data
x_val = np.load("data/x_val.npy")
cnn_preds = pd.read_csv("results/cnn_predictions.csv")["cnn_prediction"]

x_val_flat = x_val.reshape(x_val.shape[0], -1)

# Load surrogate
tree = joblib.load("models/surrogate_tree.pkl")

# Surrogate predictions
surrogate_preds = tree.predict(x_val_flat)

# Fidelity = agreement with CNN
fidelity = accuracy_score(cnn_preds, surrogate_preds)

print("Fidelity Score:", fidelity)
print("Number of nodes:", tree.tree_.node_count)
print("Tree depth:", tree.tree_.max_depth)