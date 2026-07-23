import numpy as np
import torch

def train(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total_samples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach() 

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        total_samples += labels.size(0)

    avg_loss = epoch_loss.item() / len(data_loader)
    accuracy = correct.item() / total_samples

    return avg_loss, accuracy







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