from data.utils import *
from torch.utils.data import DataLoader
from data.cifar_dataset import CIFARDataset

cifar_data = unpickle("data/cifar_training.pkl")

div_indices = [
    4 * len(cifar_data) // 5,
    4 * len(cifar_data) // 5 + len(cifar_data) // 10,
]
train_data = cifar_data[: div_indices[0]]
val_data = cifar_data[div_indices[0] : div_indices[1]]
test_data = cifar_data[div_indices[1] :]

train_dataset = CIFARDataset(train_data)
val_dataset = CIFARDataset(val_data)
test_dataset = CIFARDataset(test_data)


def rgb_to_grayscale(images):
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    grayscale = grayscale.unsqueeze(1)
    return grayscale.repeat(1, 3, 1, 1)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

gray_train_loader = DataLoader(
    rgb_to_grayscale(train_dataset), batch_size=64, shuffle=True
)
gray_val_loader = DataLoader(
    rgb_to_grayscale(val_dataset), batch_size=64, shuffle=False
)
gray_test_loader = DataLoader(
    rgb_to_grayscale(test_dataset), batch_size=64, shuffle=False
)

data_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
gray_data_loaders = {
    "train": gray_train_loader,
    "val": gray_val_loader,
    "test": gray_test_loader,
}

save_data(data_loaders, "data/cifar_image_dataloaders.pkl")
save_data(gray_data_loaders, "data/cifar_grayscale_dataloaders.pkl")
