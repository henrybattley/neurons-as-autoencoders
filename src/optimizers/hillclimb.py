import numpy as np
import torch

#dedicated hill climber for the multi-model networks

def hill_climb(model, data_loader,criterion, device,rng):
    model.to(device)
    model.train()


    delta = rng.uniform(-1,1)
    
    #50/50 chance of purturbing hidden vs output layer params
    layer_perturb = rng.random()
    if layer_perturb <= 0.5: #perturb hidden layer


        #choose a random hidden node
        selected_neuron = rng.integers(model.hidden_dim)

        params_per_neuron = (
        model.input_dim      # encoder weights
        + 1                  # encoder bias
        + model.input_dim    # decoder weights
        + model.input_dim   #decoder biases
        )


        #random parameter associated with a neuron (that is the encoding and decoding neuron connections)
        idx = rng.integers(params_per_neuron)

        if idx < model.input_dim:

            layer_module = model.hidden
            parameter_type = "weight"

            row = selected_neuron
            col = idx

        elif idx == model.input_dim:

            layer_module = model.hidden
            parameter_type = "bias"

            row = selected_neuron

        elif idx <model.input_dim +model.input_dim + 1:


            decoder_idx = idx - (model.input_dim + 1)

            layer_module = model.decoder
            parameter_type = "weight"

            row = decoder_idx        # reconstructed input dimension
            col = selected_neuron      # hidden neuron

        else:
            #decoder biases
            decoder_bias_idx = idx -(2 * model.input_dim+ 1)

            layer_module = model.decoder
            parameter_type = "bias"

            row = decoder_bias_idx

    

        """ this ise computed after we select a parameter that belongs to a certain neuron and then we'll compute the loss of what that neuron produces-- we'll compute both and see which aligns with the paper data"""
        epoch_loss = 0.0
        for inputs, labels, target_inputs in data_loader:
            inputs, labels, target_inputs = inputs.to(device), labels.to(device), target_inputs.to(device)

            #x_hat should be the output for one neuron?  -
            #x_hat, _ = model(inputs)

            #output from a single selected neuron
            x_hat_single = model.reconstruct_single(
                inputs,
                neuron=selected_neuron
            )

            #target inputs are in range [0,1] binary values since sigmoid can't output -1s
            #loss = criterion(x_hat_single, target_inputs)

            #trying different transform
            loss = criterion(x_hat_single*2-1, inputs)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)


        #taking loss of single neuron output for the layer-based encoder? --quite possibly



    else:   #set indexes to perturb output layer

        #change the loss function to be task loss        
        epoch_loss = 0.0
        for inputs, labels, target_inputs in data_loader:
            inputs, labels, target_inputs = inputs.to(device), labels.to(device), target_inputs.to(device)
            
            #outputs = model(inputs)
            #loss = criterion(outputs, labels)
     
            y = model.regress(inputs)
            loss = criterion(y, labels)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)

        
        #params in output layer
        n_ouput_w = model.output.weight.numel()
        n_output_b = model.output.bias.numel()

        total = (
            n_ouput_w
            + n_output_b
        )

        #random parameter from the output layer
        idx = rng.integers(total)

        if idx < n_ouput_w:
            #do the W perturbations..
            #define the row and col
            layer_module = model.output
            parameter_type = "weight"

            row = 0
            col = idx % model.hidden_dim


        else:
            #do the b perturbations
            #define the row

            layer_module = model.output
            parameter_type = "bias"

            row = 0


    # do the actual perturbations here based on predefined layer parameter positions
    if parameter_type == "weight":

        old = layer_module.weight[row, col].item()
        with torch.no_grad():
            layer_module.weight[row, col] += delta
    else:

        old = layer_module.bias[row].item()
        with torch.no_grad():
            layer_module.bias[row] += delta


    #now get the loss again and compare previous

    new_epoch_loss = 0.0
    for inputs, labels, target_inputs  in data_loader:
        inputs, labels, target_inputs = inputs.to(device), labels.to(device), target_inputs.to(device)
    
        #x_hat,y = model(inputs)

        #output from a single selected neuron
        #x_hat_single = model.reconstruct_single(inputs,neuron=selected_neuron)

        if layer_perturb <= 0.5:

            #loss = criterion(x_hat, target_inputs)
            #output from a single selected neuron
            x_hat_single = model.reconstruct_single(inputs,neuron=selected_neuron)
         
            #loss = criterion(x_hat_single, target_inputs)

            loss = criterion(x_hat_single*2-1, inputs)

        else:

            y = model.regress(inputs)

            loss = criterion(y, labels)

        new_epoch_loss += loss.item()

    new_avg_loss = new_epoch_loss / len(data_loader)

    if new_avg_loss > avg_loss:
        if parameter_type == 'weight':
            with torch.no_grad():
                layer_module.weight[row, col] = old
        else:
            with torch.no_grad():
                layer_module.bias[row] = old

        
        return avg_loss,layer_perturb
    
    # when perturbed parameter results in the same loss (within the fractional threshold) then chose either new or old parameter at random
    elif abs(new_avg_loss - avg_loss) < 1e-12:
        tie_break = rng.random()
        if parameter_type == 'weight': #if we previously perturbed a weight
            if tie_break >= 0.5:
                with torch.no_grad():
                    layer_module.weight[row, col] = old
  
        else: #if we previously perturbed a bias
            if tie_break < 0.5:
                with torch.no_grad():
                    layer_module.bias[row] = old

    
    return new_avg_loss,layer_perturb


"""

THIS SCALES MORE NATURALLY FOR PARAMETER TYPE:

parameter = rng.integers(tot_layer_params)

if parameter < tot_w:
"""


""" 
        row = rng.integers(model.hidden_dim)
        col = rng.integers(model.input_dim)
        
        
        tot_b= model.hidden.bias.numel()  #may use n of rows for the mlp but not for the nan i don't think- actually biases are just a vector per layer surely so we can reference them that way with one index?
        tot_w= model.hidden.weight.numel()
        tot_layer_params = tot_b + tot_w

        W_update_p = tot_w/tot_layer_params
        b_update_p = tot_b/tot_layer_params

        layer = 'hidden'
        layer_module = getattr(model, layer)"""


