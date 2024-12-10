from data.utils import *
import torch
from PIL import Image
import umap
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Load the model
model_file = "final_models/diag_2.pkl"
model = unpickle(model_file)

# Load the anomaly data
anomaly_data_file = "data/mnist_images.pkl"
anomaly_data_pts = unpickle(anomaly_data_file)["images"][:1000] / 255.0

anomaly = []
for data_pt in anomaly_data_pts:
    mean = model(data_pt.reshape(1, 3, 32, 32))[2][0]
    anomaly.append(mean.detach().numpy())

# Load the normal data
load_normal_file = "data/cifar_image_dataloaders.pkl"
loaders = unpickle(load_normal_file)

test_data = loaders["test"].dataset
test_data_pts = torch.reshape(test_data[:1000], (-1, 3, 32, 32))

test = []
for data_pt in test_data_pts:
    mean = model(data_pt.reshape(1, 3, 32, 32))[2][0]
    test.append(mean.detach().numpy())

all_data = np.array(anomaly + test)
labels = np.array([1] * len(anomaly) + [0] * len(test))

colors = ["red" if label == 1 else "blue" for label in labels]

# UMAP the data
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
embedding = reducer.fit_transform(all_data)

plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.8)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP Projection for Diag VAE")
plt.grid()

plt.scatter([], [], c="blue", label="Normal", alpha=0.8)
plt.scatter([], [], c="red", label="Anomaly", alpha=0.8)
plt.legend(loc="best")

plt.show()
