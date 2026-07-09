import numpy as np
import torch

#dedicated hill climber for the multi-model networks

def nan_hill_climb(model, data_loader,criterion, device,rng):
    model.to(device)
    model.train()


    delta = rng.uniform(-1,1)
    
    #50/50 chance of purturbing hidden vs output layer params
    layer_perturb = rng.random()
    if layer_perturb <= 0.5: #perturb hidden layer

        #params in hidden layer
        n_hidden_w = model.hidden.weight.numel()
        n_hidden_b = model.hidden.bias.numel()

        #params in decoder layer
        #n_decoder_w = model.decoder.weight.numel()
        #n_decoder_b = model.decoder.bias.numel()
        n_decoder_w = model.decoder_weight.numel()
        n_decoder_b = model.decoder_bias.numel()

        total = (
            n_hidden_w
            + n_hidden_b
            + n_decoder_w
            + n_decoder_b
        )

        #random parameter associated with the 'hidden layer' (that is the encoding and decoding neuron connections)
        idx = rng.integers(total)

        if idx < n_hidden_w:
            #index the hidden weights to subsequently be pertubed with probability n_hidden_w/total

            layer_module = model.hidden
            parameter_type = "weight"

            row = idx // model.input_dim #floor divide by the input dims to get the row of the random weight
            col = idx % model.input_dim # modulo by the input dims to get the column
            
            selected_neuron = row

        elif idx < n_hidden_w + n_hidden_b:
            #index hidden bias

            layer_module = model.hidden
            parameter_type = "bias"

            #make row relative to the layer indexing not the whole parameter space indexing
            row = idx - n_hidden_w

            selected_neuron = row


        elif idx < n_hidden_w + n_hidden_b + n_decoder_w:
            #index decoder w

            #layer_module = model.decoder
            parameter_tensor = model.decoder_weight
            #parameter_type = "weight"
            parameter_type = "decoder_weight"
            
 
            #again make indexing relative to layer 

            decoder_idx = idx - (n_hidden_w + n_hidden_b)

            row = decoder_idx // model.input_dim
            col = decoder_idx % model.input_dim

            selected_neuron = row

        else: 
            #index decoder b
 
            #layer_module = model.decoder
            parameter_tensor = model.decoder_bias
            #parameter_type = "bias"
            parameter_type = "decoder_bias"


            #row = idx - (n_hidden_w + n_hidden_b + n_decoder_w)

            #bias parameters are now of shape (input x hidden)
            decoder_idx = idx - (
                n_hidden_w
                + n_hidden_b
                + n_decoder_w
            )

            row = decoder_idx // model.input_dim
            col = decoder_idx % model.input_dim

            selected_neuron = row



        """ this is computed after we select a parameter that belongs to a certain neuron and then we'll compute the loss of what that neuron produces-- we'll compute both and see which aligns with the paper data"""
        
        epoch_loss = 0.0
        for inputs, labels, target_inputs in data_loader:
            inputs, labels, target_inputs = inputs.to(device), labels.to(device), target_inputs.to(device)

            #x_hat should be the output for one neuron?  -- this model output can be outside of these ifs anyway 
      

            #output from a single selected neuron
            x_hat_single = model.reconstruct_single(
                inputs,
                neuron=selected_neuron
            )

            #target inputs are in range [0,1] binary values since sigmoid can't output -1s
            #loss = criterion(x_hat_single, target_inputs)
            loss= criterion(x_hat_single*2-1, inputs)

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
     
            y = model(inputs)
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

    elif parameter_type =="decoder_weight":

        old = parameter_tensor[row, col].item()

        with torch.no_grad():
            parameter_tensor[row, col] += delta

    elif parameter_type =="decoder_bias":

        old = parameter_tensor[row, col].item()

        with torch.no_grad():
            parameter_tensor[row, col] += delta

    else:

        old = layer_module.bias[row].item()
        with torch.no_grad():
            layer_module.bias[row] += delta


    #now get the loss again and compare previous

    new_epoch_loss = 0.0
    for inputs, labels, target_inputs  in data_loader:
        inputs, labels, target_inputs = inputs.to(device), labels.to(device), target_inputs.to(device)
    


        if layer_perturb <= 0.5:

            #output from a single selected neuron
            x_hat_single = model.reconstruct_single(inputs,neuron=selected_neuron)
         
            #loss = criterion(x_hat_single, target_inputs)

            loss= criterion(x_hat_single*2-1, inputs)

        else:

            y = model(inputs)
            loss = criterion(y, labels)

        new_epoch_loss += loss.item()

    new_avg_loss = new_epoch_loss / len(data_loader)

    if new_avg_loss > avg_loss:
        if parameter_type == 'weight':
            with torch.no_grad():
                layer_module.weight[row, col] = old

        elif parameter_type == 'decoder_weight':
            with torch.no_grad():
                parameter_tensor[row, col] = old

        elif parameter_type == 'decoder_bias':
            with torch.no_grad():
                parameter_tensor[row, col] = old


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
        elif parameter_type in ("decoder_weight", "decoder_bias"): 
            if tie_break >= 0.5:
                with torch.no_grad():
                    parameter_tensor[row, col] = old

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