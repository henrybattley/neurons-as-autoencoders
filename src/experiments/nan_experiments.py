import os
import numpy as np
import torch, torchvision
import random


from src.models.NAN import NAN
from src.models.local_gd_nan import NAN_GD
from src.datasets.NK import NKLandscape
from src.optimizers import global_backprop
from src.optimizers import nan_hillclimb
from src.optimizers import nan_gradient_descent



#Training pipeline, when hill_climb=True training follows that as defined by Bull 
def train_NAN(  data, 
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                hill_climb=True,
                seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "test_loss": [],
    }
    

    #student's gpu is non-CUDA enabled
    device = torch.device('cpu')

    #seed randomness (already performed in notebook but now training is self-contained)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_loader = torch.utils.data.DataLoader( data,
                                                batch_size=batch_size,
                                                shuffle=False if hill_climb else True, #no need to shuffle when hill climbing since we compute activations on whole dataset (not batches)
                                                )

    #sample and retrieve shape for parameterizing model input dims
    sample_x, _,_= data[0]
    input_dim = sample_x.shape[0]

    print(f"input dims are {input_dim}")

    model = NAN(input_dim=input_dim,hidden_dim=10) # 10 nodes in the hidden layer (H of 10) as per Bull's experiments
    model.to(device)

    if hill_climb == False: 
        """ if I get time to come back to testing the multiobjective combined global loss for backprop"""
   
        criterion = torch.nn.MSELoss() #criterion is actually equal to our funky multiobjective term here I think
        criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loss = global_backprop.train(model, train_loader, criterion, optimizer, device)

        # another training epoch loop:...
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")
        training_history["train_loss"].append(train_loss)
    
    else:
        criterion = torch.nn.MSELoss() 
        criterion.to(device)

        # Training loop
        for epoch in range(n_epochs):

            nan_hillclimb.nan_hill_climb(model, train_loader,criterion, device, rng)
            encoder_loss = 0.0

            # do the reconstruction with every neuron one after the other
            for neuron in range(model.hidden_dim):

                neuron_loss = 0.0

                for inputs, _, target_inputs in train_loader:

                    inputs = inputs.to(device)
                    target_inputs = target_inputs.to(device)

                    x_hat = model.reconstruct_single(inputs, neuron)

                    #accumulate the loss
                    #neuron_loss += criterion(x_hat, target_inputs).item()
                    neuron_loss += criterion(x_hat*2-1, inputs).item()
                        

                #len of train loader is one here (but we may want batches with other data)
                neuron_loss /= len(train_loader) 

                encoder_loss += neuron_loss

            #divide by how many neurons in the hidden layer (so we average over the hidden neurons)
            encoder_loss /= model.hidden_dim

            print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {encoder_loss:.4f}")
            training_history["encoder_train_loss"].append(encoder_loss)


            epoch_loss = 0.0
            for inputs, labels, _ in train_loader:
                inputs, labels= inputs.to(device), labels.to(device)
        
                
                y = model.regress(inputs)
                loss = criterion(y, labels)

                epoch_loss += loss.item()

            avg_regression_loss = epoch_loss / len(train_loader)

            print(f"Epoch [{epoch + 1}/{n_epochs}], Regression Training Loss: {avg_regression_loss:.4f}")
            training_history["task_train_loss"].append(avg_regression_loss)


    return model, training_history


def linear_schedule_nan(data, n_epochs=100, batch_size=64, hill_climb=True, seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "test_loss": [],
    }
    
    #student's gpu is non-CUDA enabled
    device = torch.device('cpu')

    #seed randomness (already performed in notebook but now training is self-contained)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_loader = torch.utils.data.DataLoader( data,
                                                batch_size=batch_size,
                                                shuffle=False if hill_climb else True, #no need to shuffle when hill climbing since we compute activations on whole dataset (not batches)
                                                )

    #sample and retrieve shape for parameterizing model input dims
    sample_x, _,_= data[0]
    input_dim = sample_x.shape[0]

    print(f"input dims are {input_dim}")

    model = NAN(input_dim=input_dim,hidden_dim=10) # 10 nodes in the hidden layer (H of 10) as per Bull's experiments
    model.to(device)
         
    criterion = torch.nn.MSELoss() 
    criterion.to(device)

    # Training loop
    for epoch in range(n_epochs):
   
        nan_hillclimb.linear_schedule_nan_hill_climb(model, train_loader,criterion, device, rng, epoch)

        encoder_loss = 0.0

        # do the reconstruction with every neuron one after the other
        for neuron in range(model.hidden_dim):

            neuron_loss = 0.0

            for inputs, _, target_inputs in train_loader:

                inputs = inputs.to(device)
                target_inputs = target_inputs.to(device)

                x_hat = model.reconstruct_single(inputs, neuron)

                #accumulate the loss
                neuron_loss += criterion(x_hat*2-1, inputs).item()
                    

            #len of train loader is one here (but we may want batches with other data)
            neuron_loss /= len(train_loader) 

            encoder_loss += neuron_loss

        #divide by how many neurons in the hidden layer (so we average over the hidden neurons)
        encoder_loss /= model.hidden_dim

        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(encoder_loss)


        epoch_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels= inputs.to(device), labels.to(device)
     
            
            y = model.regress(inputs)
            loss = criterion(y, labels)

            epoch_loss += loss.item()

        avg_regression_loss = epoch_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Regression Training Loss: {avg_regression_loss:.4f}")
        training_history["task_train_loss"].append(avg_regression_loss) 
   
    return model, training_history



def parallel_schedule_nan(data, n_epochs=100, batch_size=64,hill_climb=True,seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "test_loss": [],
    }
    
    #student's gpu is non-CUDA enabled
    device = torch.device('cpu')

    #seed randomness (already performed in notebook but now training is self-contained)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_loader = torch.utils.data.DataLoader( data,
                                                batch_size=batch_size,
                                                shuffle=False if hill_climb else True, #no need to shuffle when hill climbing since we compute activations on whole dataset (not batches)
                                                )

    
    #sample and retrieve shape for parameterizing model input dims
    sample_x, _,_= data[0]
    input_dim = sample_x.shape[0]

    print(f"input dims are {input_dim}")

    model = NAN(input_dim=input_dim,hidden_dim=10) # 10 nodes in the hidden layer (H of 10) as per Bull's experiments
    model.to(device)
    criterion = torch.nn.MSELoss() #criterion is actually equal to our funky multiobjective term here I think
    criterion.to(device)

    # Training loop
    for epoch in range(n_epochs):
            
        nan_hillclimb.parallel_step_schedule_hill_climb(model, train_loader,criterion, device, rng)

        encoder_loss = 0.0

        # do the reconstruction with every neuron one after the other
        for neuron in range(model.hidden_dim):

            neuron_loss = 0.0

            for inputs, _, target_inputs in train_loader:

                inputs = inputs.to(device)
                target_inputs = target_inputs.to(device)

                x_hat = model.reconstruct_single(inputs, neuron)

                #accumulate the loss
                neuron_loss += criterion(x_hat*2-1, inputs).item()
                    

            #len of train loader is one here (but we may want batches with other data)
            neuron_loss /= len(train_loader) 

            encoder_loss += neuron_loss

        #divide by how many neurons in the hidden layer (so we average over the hidden neurons)
        encoder_loss /= model.hidden_dim

        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(encoder_loss)


        epoch_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels= inputs.to(device), labels.to(device)
     
            
            y = model.regress(inputs)
            loss = criterion(y, labels)

            epoch_loss += loss.item()

        avg_regression_loss = epoch_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Regression Training Loss: {avg_regression_loss:.4f}")
        training_history["task_train_loss"].append(avg_regression_loss) 
   
    return model, training_history



#training and testing pipeline with standard 50/50 stochastic training split as described by Bull
def train_NAN_and_test(  data, 
                test_data,
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                hill_climb=True,
                seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "encoder_test_loss": [],
    "task_test_loss": [],
    }
    

    #student's gpu is non-CUDA enabled
    device = torch.device('cpu')

    #seed randomness (already performed in notebook but now training is self-contained)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)


    train_loader = torch.utils.data.DataLoader( data,
                                                batch_size=batch_size,
                                                shuffle=False if hill_climb else True, #no need to shuffle when hill climbing since we compute activations on whole dataset (not batches)
                                                )

    test_loader = torch.utils.data.DataLoader(  test_data,
                                                batch_size=batch_size,
                                                shuffle=False if hill_climb else True, 
                                                ) 
    

    #sample and retrieve shape for parameterizing model input dims
    sample_x, _,_= data[0]
    input_dim = sample_x.shape[0]

    print(f"input dims are {input_dim}")


    model = NAN(input_dim=input_dim,hidden_dim=10) # 10 nodes in the hidden layer (H of 10) as per Bull's experiments
    model.to(device)
    criterion = torch.nn.MSELoss() 
    criterion.to(device)


    # Training loop
    for epoch in range(n_epochs):
            

        nan_hillclimb.nan_hill_climb(model, train_loader,criterion, device,rng)

        encoder_loss = 0.0

        """ not that interested in seeing the encoding error 
            # do the reconstruction with every neuron one after the other
            for neuron in range(model.hidden_dim):

                neuron_loss = 0.0

                #instead report error on the test data
                for inputs, _, target_inputs in test_loader:

                    inputs = inputs.to(device)
                    target_inputs = target_inputs.to(device)

                    x_hat = model.reconstruct_single(inputs, neuron)

                    #accumulate the loss
                    #neuron_loss += criterion(x_hat, target_inputs).item()
                    neuron_loss += criterion(x_hat*2-1, inputs).item()
                    

                #len of train loader is one here (but we may want batches with other data)
                neuron_loss /= len(train_loader) 

                encoder_loss += neuron_loss

            #divide by how many neurons in the hidden layer (so we average over the hidden neurons)
            encoder_loss /= model.hidden_dim

            print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {encoder_loss:.4f}")
            training_history["encoder_train_loss"].append(encoder_loss)
            """

        epoch_loss = 0.0
        for inputs, labels, _ in test_loader:
            inputs, labels= inputs.to(device), labels.to(device)
     
            
            y = model.regress(inputs)
            loss = criterion(y, labels)

            epoch_loss += loss.item()

        avg_regression_loss = epoch_loss / len(test_loader)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Regression Test Loss: {avg_regression_loss:.4f}")
        training_history["task_test_loss"].append(avg_regression_loss)


    return model, training_history





def train_local_gradient_NAN(  data, 
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "test_loss": [],
    }
    
    #student's gpu is non-CUDA enabled
    device = torch.device('cpu')

    #seed randomness (already performed in notebook but now training is self-contained)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)


    train_loader = torch.utils.data.DataLoader( data,batch_size=batch_size,shuffle= True,)

    
    #sample and retrieve shape for parameterizing model input dims
    sample_x, _,_= data[0]
    input_dim = sample_x.shape[0]

    print(f"input dims are {input_dim}")


    model = NAN_GD(input_dim=input_dim,hidden_dim=10) # 10 nodes in the hidden layer (H of 10) as per Bull's experiments

    model.to(device)
    criterion = torch.nn.MSELoss() 
    criterion.to(device)


    hidden_optimizers = []

    for neuron in range(model.hidden_dim):

        hidden_optimizers.append(

            torch.optim.Adam(
            [
                model.encoder_weights[neuron],
                model.encoder_biases[neuron],
                model.decoder_weights[neuron],
                model.decoder_biases[neuron],
            ],
            lr=learning_rate)

        )
    

    output_optimizer = torch.optim.Adam(model.output.parameters(), lr=learning_rate)


    # Training loop
    for epoch in range(n_epochs):

        encoder_loss=0.0

        for neuron in range(model.hidden_dim):

            train_loss = nan_gradient_descent.train_hidden_layer(model, train_loader, criterion, hidden_optimizers[neuron], device, neuron)

            encoder_loss += train_loss

        #divide by how many neurons in the hidden layer (so we average over the hidden neurons)
        encoder_loss /= model.hidden_dim
        
        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(encoder_loss)

        epoch_loss = 0.0
    
        train_loss = nan_gradient_descent.train_output_layer(model, train_loader, criterion, output_optimizer, device)


        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {train_loss:.4f}")
        training_history["task_train_loss"].append(train_loss)


    return model, training_history
        
 




    
