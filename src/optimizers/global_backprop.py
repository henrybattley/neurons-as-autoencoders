import numpy as np
import torch
import time


def train(model, data_loader, criterion, optimizer, device):
    model.to(device)
    model.train()

    epoch_loss = 0.0
    for inputs, labels in data_loader:


        t0 = time.perf_counter()

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        optimizer.zero_grad()
        loss.backward()

        torch.cuda.synchronize()
        t3 = time.perf_counter()

        optimizer.step()

        torch.cuda.synchronize()
        t4 = time.perf_counter()

        print(
            f"copy={t1-t0:.4f} "
            f"forward={t2-t1:.4f} "
            f"backward={t3-t2:.4f} "
            f"step={t4-t3:.4f}"
        )

        break

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)
    return avg_loss

""" may come in useful later for test set? 
def eval(model, data_loader, criterion, device):
    model.to(device)
    model.eval()

    targets = []
    predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #_, predicted = torch.max(outputs.data, 1)

            targets.extend(labels.cpu().numpy())
            #predictions.extend(predicted.cpu().numpy())
            total_loss += loss.item()

    #accuracy = (np.array(predictions) == np.array(targets)).mean()
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss

    #return avg_loss, accuracy
    """