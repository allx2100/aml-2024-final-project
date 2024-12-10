import numpy as np
import struct
import torch
import pickle


def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid magic number for image file")

        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        return image_data.reshape(num_images, rows, cols)


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic, _ = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number for label file")

        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        return label_data


def expand_to_32x32(images, padding_value=0):
    num_images, orig_height, orig_width = images.shape
    new_height, new_width = 32, 32
    expanded_images = torch.full(
        (num_images, new_height, new_width), padding_value, dtype=torch.uint8
    )

    start_x = (new_width - orig_width) // 2
    start_y = (new_height - orig_height) // 2
    expanded_images[
        :, start_y : start_y + orig_height, start_x : start_x + orig_width
    ] = torch.tensor(images)

    return expanded_images


def convert_to_rgb(images_tensor):
    num_images = images_tensor.shape[0]
    images_rgb = torch.stack(
        [torch.tensor(np.stack([image.numpy()] * 3, axis=0)) for image in images_tensor]
    )
    return images_rgb


train_images_path = "data/MNIST/train-images.idx3-ubyte"
train_labels_path = "data/MNIST/train-labels.idx1-ubyte"
test_images_path = "data/MNIST/t10k-images.idx3-ubyte"
test_labels_path = "data/MNIST/t10k-labels.idx1-ubyte"

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

train_data = {"images": train_images_rgb_tensor, "labels": train_labels}

with open("data/mnist_images.pkl", "wb") as f:
    pickle.dump(train_data, f)
