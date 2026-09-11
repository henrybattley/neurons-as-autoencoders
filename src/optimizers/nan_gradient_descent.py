import numpy as np
import torch



def train_hidden_layer(model, data_loader, criterion, optimizer, device, selected_neuron):

    model.to(device)
    model.train()
   
    # now for the chosen hidden neuron, compute the reconstruction error for the neuron's attempt at reconstructing its input
    epoch_loss = 0.0
    for inputs, _, _ in data_loader:
        inputs = inputs.to(device)

        optimizer.zero_grad()

        #output from a single selected neuron
        x_hat_single = model.reconstruct_single(inputs,neuron=selected_neuron)

        #decoder ouput needs to be scaled to [-1,1] range
        loss = criterion(x_hat_single*2-1, inputs)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)

    return avg_loss




def train_output_layer(model, data_loader, criterion, optimizer, device):
    model.to(device)
    model.train()

    epoch_loss = 0.0
    for inputs, labels, _ in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        y= model.regress(inputs)

        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)

    return avg_loss







