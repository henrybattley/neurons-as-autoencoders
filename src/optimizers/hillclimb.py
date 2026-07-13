import numpy as np
import torch

#dedicated hill climber for the layer based autoencoder network

def hill_climb(model, data_loader,criterion, device,rng):

    model.to(device)
    model.train()

    #random perturbation to be added to the parameters
    delta = rng.uniform(-1,1)
    
    #50/50 chance of purturbing hidden vs output layer params
    layer_perturb = rng.random()

    if layer_perturb <= 0.5: #perturb hidden layer

        #choose a random hidden node
        selected_neuron = rng.integers(model.hidden_dim)

        #quantify the possible params for every node 
        params_per_neuron = (
        model.input_dim      # encoder weights (one weight per input dim)
        + 1                  # encoder bias (one bias per node)
        + model.input_dim    # decoder weights (one weight per input dim)
        + model.input_dim   #decoder biases (biases are shared within the decoder layer)
        )


        #random parameter associated with selected neuron (that is the encoding and decoding neuron connections)
        idx = rng.integers(params_per_neuron)

        if idx < model.input_dim: # index the encoder weights

            layer_module = model.hidden
            parameter_type = "weight"

            row = selected_neuron #rows are neurons within the encoding connections
            col = idx #specific weight is the random idx referring to a column within the encoding connection matrix

        elif idx == model.input_dim: #index the encoder bias 

            layer_module = model.hidden
            parameter_type = "bias"

            row = selected_neuron #encoder biases are within a vector, we reference each by their corresponding neuron

        elif idx < model.input_dim + model.input_dim + 1: # index decoder weights

            #make global parameter index relative to decoder layer weights
            decoder_idx = idx - (model.input_dim + 1)

            layer_module = model.decoder
            parameter_type = "weight"

            row = decoder_idx   # specific weight is the random idx referring to a row within the decoding connection matrix
            col = selected_neuron #columns are neurons within the decoding connections

        else:
            #decoder biases
            #make parameter reference relative to decoder layer biases
            decoder_bias_idx = idx -(2 * model.input_dim+ 1)

            layer_module = model.decoder
            parameter_type = "bias"

            #again biases are within a vector
            row = decoder_bias_idx


        # now for the chosen hidden neuron, compute the reconstruction error for the neuron's attempt at reconstructing its input
        epoch_loss = 0.0
        for inputs, _, _ in data_loader:
            inputs = inputs.to(device)

            #output from a single selected neuron
            x_hat_single = model.reconstruct_single(inputs,neuron=selected_neuron)

            #decoder ouput needs to be scaled to [-1,1] range
            loss = criterion(x_hat_single*2-1, inputs)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)



    else:  #(when layer_perturb > 0.5) set indexes for output layer parameter perturbation

        #params in output layer (need not be referenced by a single neuron)
        n_ouput_w = model.output.weight.numel()
        n_output_b = model.output.bias.numel()

        total = (n_ouput_w + n_output_b)

        #random parameter from the output layer
        idx = rng.integers(total)

        if idx < n_ouput_w: #index the output weights
            layer_module = model.output
            parameter_type = "weight"

            row = 0 #weights are within a vector
            col = idx # weight selected corresponds to the index within [0,<hidden_dim]


        else: #index the output bias
            layer_module = model.output
            parameter_type = "bias"

            row = 0 #only one bias


        #change the loss function to be task loss        
        epoch_loss = 0.0
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            y = model.regress(inputs)
            loss = criterion(y, labels)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)


    # do the actual perturbations here based on predefined layer parameter positions
    if parameter_type == "weight":

        #store the previous weight in memory
        old = layer_module.weight[row, col].item()
        with torch.no_grad():
            layer_module.weight[row, col] += delta
    else:
        #store the previous bias in memory
        old = layer_module.bias[row].item()
        with torch.no_grad():
            layer_module.bias[row] += delta


    #now get the new loss, and compare previous
    new_epoch_loss = 0.0

    if layer_perturb <= 0.5: #compute new autoencoder loss

        #new_epoch_loss = 0.0
        for inputs, _,_  in data_loader:
            inputs = inputs.to(device)

            #new output from a single selected neuron
            x_hat_single = model.reconstruct_single(inputs,neuron=selected_neuron)
    
            loss = criterion(x_hat_single*2-1, inputs)

            new_epoch_loss += loss.item()


        new_avg_loss = new_epoch_loss / len(data_loader)


    else: #compute new task loss
        for inputs, labels,_  in data_loader:
            inputs,labels = inputs.to(device), labels.to(device)

            y = model.regress(inputs)

            loss = criterion(y, labels)

            new_epoch_loss += loss.item()

        new_avg_loss = new_epoch_loss / len(data_loader)

    if new_avg_loss > avg_loss:
        if parameter_type == 'weight':
            with torch.no_grad():
                layer_module.weight[row, col] = old #restore previous weight if the weight perturbation caused increase in error
        else:
            with torch.no_grad():
                layer_module.bias[row] = old #restore previous bias if the bias perturbation caused increase in error

        
        return avg_loss,layer_perturb
    
    # when perturbed parameter results in the same loss (within the fractional threshold) then chose either new or old parameter at random
    elif abs(new_avg_loss - avg_loss) < 1e-12:
        tie_break = rng.random()
        if parameter_type == 'weight': #if we previously perturbed a weight
            if tie_break >= 0.5: #restore previous parameter, otherwise by default the new parameter is kept
                with torch.no_grad():
                    layer_module.weight[row, col] = old
  
        else: #if we previously perturbed a bias
            if tie_break < 0.5:
                with torch.no_grad():
                    layer_module.bias[row] = old
        






