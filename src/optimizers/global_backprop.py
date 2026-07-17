import numpy as np
import torch

""" 
def train(model, data_loader, criterion, optimizer, device):

    epoch_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device,non_blocking=True)

        optimizer.zero_grad()

        #degub
        #print(inputs.device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Classification accuracy
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = epoch_loss / len(data_loader)
    classification_accuracy = 100.0 * correct / total
    
    return avg_loss, classification_accuracy"""

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

        # 1. Keep everything on the GPU as a raw tensor during the loop
        epoch_loss += loss.detach() 

        # 2. Calculate correctness entirely on the GPU
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        total_samples += labels.size(0)

    # 3. ONLY call .item() ONCE at the very end of the entire epoch!
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