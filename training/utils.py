import torch


def train(data_loader, model, loss_fn, optimizer, train_loss, recon_loss, kld_list):
    size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    running_loss = 0.0
    running_recon = 0.0
    running_kld = 0.0

    model.train()
    for i, data in enumerate(data_loader):
        inputs = data
        labels = data

        optimizer.zero_grad()
        outputs = model(inputs)

        total_loss = loss_fn(outputs)
        loss = total_loss["loss"]
        r_loss = total_loss["Reconstruction_Loss"]
        KLD = total_loss["KLD"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
            vloss = loss_fn(voutputs)["loss"]
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
