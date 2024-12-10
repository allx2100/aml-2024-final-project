from data.utils import *
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

# Load in model, change the model_file to test different models
model_file = "final_models/diag_2.pkl"
model = unpickle(model_file)

# Load in normal (test) data
load_train_file = "data/cifar_grayscale_dataloaders.pkl"
loaders = unpickle(load_train_file)
test_data = loaders["test"].dataset
test_data_pts = torch.reshape(test_data[:100], (-1, 3, 32, 32))

# Load in anomaly data
anomaly_data_file = "data/cifar_image_dataloaders.pkl"
anomaly_data = unpickle(anomaly_data_file)["train"].dataset

anomaly_data_pts = torch.reshape(anomaly_data[:100], (-1, 3, 32, 32))

true_labels = np.array([1] * len(anomaly_data_pts) + [0] * len(test_data_pts))
scores = np.array(
    [
        F.mse_loss(model.reconstruct(pt.reshape(1, 3, 32, 32))[0], pt).item()
        for pt in anomaly_data_pts
    ]
    + [
        F.mse_loss(model.reconstruct(pt.reshape(1, 3, 32, 32))[0], pt).item()
        for pt in test_data_pts
    ]
)

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Diag VAE")
plt.legend(loc="lower right")
plt.grid()
plt.show()

print(f"AUC (Area Under Curve): {roc_auc}")
