import torch
import dill
from models.diag_vae import DiagVAE
from data.utils import *

def train(data_loader, model, loss_fn, optimizer, train_loss, recon_loss, kld_list):
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    running_loss = 0.
    running_recon = 0.
    running_kld = 0.
    
    model.train()
    for i, data in enumerate(data_loader):
        inputs = data
        labels = data
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        total_loss = loss_fn(outputs)
        loss = total_loss['loss']
        r_loss = total_loss['Reconstruction_Loss']
        KLD = total_loss['KLD']

        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        running_recon += r_loss.item()
        running_kld += KLD.item()
        
    train_loss.append(running_loss / size * batch_size)
    recon_loss.append(running_recon / size * batch_size)
    kld_list.append(running_kld / size * batch_size)
    
    return running_loss / size

def validate(data_loader, model, loss_fn, val_loss):
    running_vloss = 0
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(data_loader):
            voutputs = model(vdata)
            vloss = loss_fn(voutputs)['loss']
            running_vloss += vloss.item()

    val_loss.append(running_vloss / size * batch_size)
    return running_vloss / size * batch_size

def test(data_loader, model):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = data
            outputs = model(inputs)
            preds.append(outputs)
    return preds

with open('data/cifar_dataloaders.pkl', 'rb') as f:
    data_loaders = dill.load(f)

train_loader = data_loaders['train']
val_loader = data_loaders['val']
test_loader = data_loaders['test']

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = DiagVAE(3072, 10, kld_weight=1e-6)
loss_fn = model.loss_function
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

train_loss = []
recon_loss = []
kld = []
val_loss = []

print("Epoch\tTrain Loss\tRecon Loss\tKL Diverg\tVal Loss")
for i in range(300):
    train(train_loader, model, loss_fn, optimizer, train_loss, recon_loss, kld)
    validate(val_loader, model, loss_fn, val_loss)
    print(f"{i+1} ", f"\t{train_loss[-1]:>7f}", f"\t{recon_loss[-1]:>7f}", \
          f"\t{kld[-1]:>7f}", f"\t{val_loss[-1]:>7f}")
    if (i+1) % 20 == 0:
        save_data(model, f'saved_models/test_model_{i+1}.pkl')