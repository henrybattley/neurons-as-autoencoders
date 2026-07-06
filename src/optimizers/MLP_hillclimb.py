import numpy as np
import torch
import random


def mlp_hill_climb(model, data_loader, criterion, device,rng):
    model.to(device)
    model.train()


    epoch_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
    

        outputs = model(inputs)
        loss = criterion(outputs, labels)


        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)

    delta = rng.uniform(-1,1)
    
    #50/50 chance of purturbing hidden vs output layer params
    if rng.random() <= 0.5: #perturb hidden layer

        row = rng.integers(model.hidden_dim)
        col = rng.integers(model.input_dim)
        tot_b= model.hidden.bias.numel()  #may use n of rows for the mlp but not for the nan i don't think- actually biases are just a vector per layer surely so we can reference them that way with one index?
        tot_w= model.hidden.weight.numel()
        tot_layer_params = tot_b + tot_w

        W_update_p = tot_w/tot_layer_params
        b_update_p = tot_b/tot_layer_params

        layer = 'hidden'
        layer_module = getattr(model, layer)


    else:   #set indexes to perturb output layer

        row = rng.integers(1)
        col = rng.integers(model.hidden_dim)

        tot_b= model.output.bias.numel() 
        tot_w= model.output.weight.numel()
        tot_layer_params = tot_b + tot_w

        layer = 'output'
        layer_module = getattr(model, layer)

    param_type = rng.random()

    if param_type <= tot_w/tot_layer_params:
        #do the w updates
        #old = model.f"{layer}"[row,col]
        #model.f"{layer}".weight[row,col] += delta

        #layer_module = getattr(model, layer)
        old = layer_module.weight[row, col].item()
        with torch.no_grad():
            layer_module.weight[row, col] += delta

    else:
        #do the b updates
        #old = model.f"{layer}".bias[row]
        #model.f"{layer}".bias[row] += delta

        #is this correct?
        #layer_module = getattr(model, layer)
        old = layer_module.bias[row].item()
        with torch.no_grad():
            layer_module.bias[row] += delta


    #now get the loss again and compare previous

    new_epoch_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
    

        outputs = model(inputs)
        loss = criterion(outputs, labels)


        new_epoch_loss += loss.item()

    new_avg_loss = new_epoch_loss / len(data_loader)

    if new_avg_loss > avg_loss:
        if param_type <= tot_w/tot_layer_params:
            #model.f"{layer}".weight[row,col]=old
            with torch.no_grad():
                layer_module.weight[row, col] = old
        else:
            #model.f"{layer}".bias[row]=old
            with torch.no_grad():
                layer_module.bias[row] = old

        
        return avg_loss
    
    # COME BACK TO THIS THRESHOLD AND THINK ABOUT
    elif abs(new_avg_loss - avg_loss) < 1e-12:
        tie_break = rng.random()
        if param_type <= tot_w/tot_layer_params:
            if tie_break >= 0.5:
                #model.f"{layer}".weight[row,col]=old
                with torch.no_grad():
                    layer_module.weight[row, col] = old
  
        else:
            if tie_break < 0.5:
                #model.f"{layer}".bias[row]=old
                with torch.no_grad():
                    layer_module.bias[row] = old

    
    return new_avg_loss


"""

THIS SCALES MORE NATURALLY FOR PARAMETER TYPE:

parameter = rng.integers(tot_layer_params)

if parameter < tot_w:
"""


