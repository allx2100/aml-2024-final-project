from data.utils import *
import torch

cifar_data_dir = 'data/cifar-10-batches-py/data.pkl'
cifar_data = unpickle(cifar_data_dir)

images = torch.tensor(cifar_data['images'] / 255.0, dtype=torch.float32)
labels = torch.tensor(cifar_data['labels'], dtype=torch.long)

image_groups = {}

for i in range(10):
    image_groups[i] = images[labels == i]

selected_images = []

for i in range(10):
    group_images = image_groups[i]
    # num_images = group_images.size(0)
    
    # indices = torch.randperm(num_images)[:1000]
    selected_images.append(group_images)

cifar_training = torch.cat(selected_images, dim=0)
indices = torch.randperm(len(cifar_training))
cifar_training = cifar_training[indices]

save_data(cifar_training, 'data/cifar_training.pkl')
