import numpy as np
import struct
import torch

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read magic number and dimensions
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError('Invalid magic number for image file')
        # Read the image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        return image_data.reshape(num_images, rows, cols)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read magic number and number of items
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError('Invalid magic number for label file')
        # Read the label data
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        return label_data

def expand_to_32x32(images, padding_value=0):
    """Expand the given images from 28x28 to 32x32 by adding padding."""
    num_images, orig_height, orig_width = images.shape
    new_height, new_width = 32, 32
    expanded_images = torch.full((num_images, new_height, new_width), padding_value, dtype=torch.uint8)
    
    # Center the 28x28 image in the 32x32 canvas
    start_x = (new_width - orig_width) // 2
    start_y = (new_height - orig_height) // 2
    expanded_images[:, start_y:start_y + orig_height, start_x:start_x + orig_width] = torch.tensor(images)
    
    return expanded_images

def convert_to_rgb(images_tensor):
    num_images = images_tensor.shape[0]
    images_rgb = torch.stack([
        torch.tensor(np.stack([image.numpy()] * 3, axis=0))  # Create 3-channel and move channels to the first dimension
        for image in images_tensor
    ])
    return images_rgb

# Paths to your MNIST files (update these paths as necessary)
train_images_path = 'data/MNIST/raw/train-images-idx3-ubyte'
train_labels_path = 'data/MNIST/raw/train-labels-idx1-ubyte'
test_images_path = 'data/MNIST/raw/t10k-images-idx3-ubyte'
test_labels_path = 'data/MNIST/raw/t10k-labels-idx1-ubyte'

# Load the data
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

train_images_tensor = expand_to_32x32(train_images)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_images_tensor = expand_to_32x32(test_images)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

train_images_rgb_tensor = convert_to_rgb(train_images_tensor)
test_images_rgb_tensor = convert_to_rgb(test_images_tensor)

def flatten_images(images_tensor):
    num_images = images_tensor.shape[0]
    flattened_images = images_tensor.view(num_images, -1)  # Flatten to (num_images, 3072)
    return flattened_images

# Flatten both train and test images
train_images_flattened = flatten_images(train_images_rgb_tensor)
test_images_flattened = flatten_images(test_images_rgb_tensor)

print('Train images RGB tensor shape:', train_images_rgb_tensor.shape)
print('Test images RGB tensor shape:', test_images_rgb_tensor.shape)

import pickle

# Create a dictionary containing the training images and labels
train_data = {
    'images': train_images_flattened,
    'labels': train_labels_tensor
}

# Save the dictionary as a pickle file
with open('mnist_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
