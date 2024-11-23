from data.utils import *
from torch.utils.data import DataLoader
from data.cifar_dataset import CIFARDataset

cifar_data = unpickle('data/cifar_training.pkl')

div_indices = [4 * len(cifar_data) // 5, 4 * len(cifar_data) // 5 + len(cifar_data) // 10]
train_data = cifar_data[:div_indices[0]]
val_data = cifar_data[div_indices[0]:div_indices[1]]
test_data = cifar_data[div_indices[1]:]

train_dataset = CIFARDataset(train_data)
val_dataset = CIFARDataset(val_data)
test_dataset = CIFARDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

data_loaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}

save_data(data_loaders, 'data/cifar_dataloaders.pkl')
