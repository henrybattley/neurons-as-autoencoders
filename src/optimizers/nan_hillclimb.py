import numpy as np
import torch

"""dedicated hill climber for the neuron based autoencoder network as described by Bull 
"""
def nan_hill_climb(model, data_loader,criterion, device,rng):

    model.to(device)
    model.train()

    #random perturbation to be added to the parameters
    delta = rng.uniform(-1,1)
    
    #50/50 chance of purturbing hidden vs output layer params
    layer_perturb = rng.random()


    if layer_perturb <= 0.5: #perturb hidden layer

        #choose a random hidden node
        selected_neuron = rng.integers(model.hidden_dim)

        #measure mse of selected neuron's attempt to reconstruct its input
        avg_loss = get_loss(type="encoder",
                            model=model,
                            data_loader=data_loader,
                            device=device,
                            criterion=criterion,
                            selected_neuron=selected_neuron)
        
        #select at random a parameter associated with that neuron 
        row, col, parameter_type, parameter_tensor,layer_module = get_random_hidden_param(model,selected_neuron,rng)

        #perturb the parameter at the defined index and return the parameter value prior to the perturbation
        old_parameter = perturb(parameter_type,parameter_tensor,layer_module,row,col,delta)

        #measure mse now that perturbation has occured
        new_avg_loss = get_loss(type="encoder",
                                model=model,
                                data_loader=data_loader,
                                device=device,
                                criterion=criterion,
                                selected_neuron=selected_neuron)

        #when parameter update impedes loss, then we restore previous parameter
        restore_if_necessary(new_avg_loss,
                             avg_loss,
                             parameter_type,
                             layer_module,
                             parameter_tensor,
                             row,
                             col,
                             old_parameter, 
                             rng)



    else:  #(when layer_perturb > 0.5) set indexes for output layer parameter perturbation

        #no need to select a neuron within the output layer
        selected_neuron=None
        
        #measure mse for regression task
        avg_loss = get_loss(type="regression",
                            model=model,
                            data_loader=data_loader,
                            device=device,
                            criterion=criterion,
                            selected_neuron=selected_neuron)

        #get random parameter position for perturbation
        row,col,parameter_type,layer_module = get_random_output_parameter(model,rng)

        #perform perturbation and get previous parameter
        old_parameter = perturb(parameter_type=parameter_type,
                                parameter_tensor=None,
                                layer_module=layer_module,
                                row=row,
                                col=col,
                                delta=delta)

        #measure new regression loss
        new_avg_loss = get_loss(type="regression",
                                model=model,
                                data_loader=data_loader,
                                device=device,
                                criterion=criterion,
                                selected_neuron=selected_neuron)

        restore_if_necessary(new_avg_loss=new_avg_loss,
                             avg_loss=avg_loss,
                             parameter_type=parameter_type,
                             layer_module=layer_module,
                             parameter_tensor=None,
                             row=row,
                             col=col,
                             old=old_parameter, 
                             rng=rng)


""" linear schedule for the neuron based autoencoder network 
"""
def linear_schedule_nan_hill_climb(model, data_loader,criterion, device, rng, epoch):

    model.to(device)
    model.train()

    #random perturbation to be added to the parameters
    delta = rng.uniform(-1,1)

    parameter_tensor=0
    layer_module=0
    row=0
    col=0

    if epoch <= 5000: #perturb hidden layer


        #choose a random hidden node
        selected_neuron = rng.integers(model.hidden_dim)


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


        #quantify the possible params for every node 
        params_per_neuron = (
        model.input_dim      # encoder weights (one weight per input dim)
        + 1                  # encoder bias (one bias per node)
        + model.input_dim    # decoder weights (one weight per input dim)
        + model.input_dim   #decoder biases (a bias per input dim)
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

            #layer_module = model.decoder

            parameter_tensor = model.decoder_weight
            parameter_type = "decoder_weight"

            row = selected_neuron  #rows correspond to neurons within the decoding connections
            col = decoder_idx # specific weight is the random idx referring to a col within the decoding connection matrix 

        else:
            #decoder biases
            #make parameter reference relative to decoder layer biases
            decoder_bias_idx = idx -(2 * model.input_dim+ 1)

            #layer_module = model.decoder

            parameter_tensor = model.decoder_bias
            parameter_type = "decoder_bias"

            #again biases are within a vector
            #row = decoder_bias_idx

            row = selected_neuron  #rows correspond to neurons within the decoding connections
            col = decoder_bias_idx # specific bias is the random idx referring to a col within the decoding connection matrix


        #perturb the random parameter from the chosen node
        old = perturb(parameter_type,parameter_tensor,layer_module,row,col,delta)


        #now get the new loss, and compare previous
        new_epoch_loss = 0.0

        #new_epoch_loss = 0.0
        for inputs, _,_  in data_loader:
            inputs = inputs.to(device)

            #new output from a single selected neuron
            x_hat_single = model.reconstruct_single(inputs,neuron=selected_neuron)
    
            loss = criterion(x_hat_single*2-1, inputs)

            new_epoch_loss += loss.item()


        new_avg_loss = new_epoch_loss / len(data_loader)


        restore_if_necessary(new_avg_loss,avg_loss,parameter_type,layer_module,parameter_tensor,row,col,old, rng)
  




    #for remainder of epochs train the output layer

    elif epoch >5000:  #(when layer_perturb > 0.5) set indexes for output layer parameter perturbation


        #change the loss function to be task loss        
        epoch_loss = 0.0
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            y = model.regress(inputs)
            loss = criterion(y, labels)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)


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

        #perturb a random param of the chosen node from the output layer 
        old = perturb(parameter_type,parameter_tensor,layer_module,row,col,delta)

        #now get the new loss, and compare previous
        new_epoch_loss = 0.0

        for inputs, labels,_  in data_loader:
            inputs,labels = inputs.to(device), labels.to(device)

            y = model.regress(inputs)

            loss = criterion(y, labels)

            new_epoch_loss += loss.item()

        new_avg_loss = new_epoch_loss / len(data_loader)


        restore_if_necessary(new_avg_loss,avg_loss,parameter_type,layer_module,parameter_tensor,row,col,old, rng)




def get_random_hidden_param(model, selected_neuron, rng,): 


    #quantify the possible params for every node 
    params_per_neuron = (
    model.input_dim      # encoder weights (one weight per input dim)
    + 1                  # encoder bias (one bias per node)
    + model.input_dim    # decoder weights (one weight per input dim)
    + model.input_dim   #decoder biases (a bias per input dim)
    )


    #random parameter associated with selected neuron (that is the encoding and decoding neuron connections)
    idx = rng.integers(params_per_neuron)

    if idx < model.input_dim: # index the encoder weights

        layer_module = model.hidden
        parameter_type = "weight"

        row = selected_neuron #rows are neurons within the encoding connections
        col = idx #specific weight is the random idx referring to a column within the encoding connection matrix

        #unused
        parameter_tensor=None

    elif idx == model.input_dim: #index the encoder bias 

        layer_module = model.hidden
        parameter_type = "bias"

        row = selected_neuron #encoder biases are within a vector, we reference each by their corresponding neuron

        #unused
        parameter_tensor=None
        col=None

    elif idx < model.input_dim + model.input_dim + 1: # index decoder weights

        #make global parameter index relative to decoder layer weights
        decoder_idx = idx - (model.input_dim + 1)

        #layer_module = model.decoder

        parameter_tensor = model.decoder_weight
        parameter_type = "decoder_weight"

        row = selected_neuron  #rows correspond to neurons within the decoding connections
        col = decoder_idx # specific weight is the random idx referring to a col within the decoding connection matrix 

        #unused
        layer_module=None

    else: #decoder biases
        #make parameter reference relative to decoder layer biases
        decoder_bias_idx = idx -(2 * model.input_dim+ 1)

        parameter_tensor = model.decoder_bias
        parameter_type = "decoder_bias"

        row = selected_neuron  #rows correspond to neurons within the decoding connections
        col = decoder_bias_idx # specific bias is the random idx referring to a col within the decoding connection matrix

        #unused
        layer_module=None

    return row, col, parameter_type, parameter_tensor, layer_module


def get_random_output_parameter(model,rng): 
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

            #unused:
            col=None

        return row,col,parameter_type,layer_module




def perturb(parameter_type,parameter_tensor,layer_module,row,col,delta):

        # do the actual perturbations here based on predefined layer parameter positions
        if parameter_type == "weight":

            #store the previous weight in memory
            old = layer_module.weight[row, col].item()
            with torch.no_grad():
                layer_module.weight[row, col] += delta

        elif parameter_type == "decoder_weight":
            old = parameter_tensor[row, col].item()
            with torch.no_grad():
                parameter_tensor[row, col] += delta
            
        elif parameter_type == "decoder_bias":
            old = parameter_tensor[row, col].item()
            with torch.no_grad():
                parameter_tensor[row, col] += delta

        else: #hidden layer bias
            #store the previous bias in memory
            old = layer_module.bias[row].item()
            with torch.no_grad():
                layer_module.bias[row] += delta

        return old


def get_loss(type,model,data_loader,device,criterion, selected_neuron):

    if type == "encoder":
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


    elif type =="regression":

        #change the loss function to be task loss        
        epoch_loss = 0.0
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            y = model.regress(inputs)
            loss = criterion(y, labels)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)

    return avg_loss



def restore_if_necessary(new_avg_loss,avg_loss,parameter_type,layer_module,parameter_tensor,row,col,old, rng):

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

            else: #hidden bias restore
                with torch.no_grad():
                    layer_module.bias[row] = old

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
        


        

