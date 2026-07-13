import os
import numpy as np
import torch, torchvision
import random


from src.models.ANN import ANN
from src.datasets.NK import NKLandscape
from src.optimizers import mlp_backprop
from src.optimizers import hillclimb


#training pipeline with default parameters
def train_ANN(  NK_data_train, 
                #NK_data_test,
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                USE_GPU=False,
                fold_id=None,
                metrics_dict=None,
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


    train_loader = torch.utils.data.DataLoader( NK_data_train,
                                                batch_size=batch_size,
                                                shuffle=False if hill_climb else True, #no need to shuffle when hill climbing since we compute activations on whole dataset (not batches)
                                                )
    """
    test_loader = torch.utils.data.DataLoader(  NK_data_test,
                                                batch_size=batch_size,
                                                shuffle=False if hill_climb else True, 
                                                ) """
    

    #sample and retrieve shape for parameterizing model input dims
    sample_x, _,_= NK_data_train[0]
    input_dim = sample_x.shape[0]

    print(f"input dims are {input_dim}")


    model = ANN(input_dim=input_dim,hidden_dim=10) # 10 nodes in the hidden layer (H of 10) as per Bull's experiments

    model.to(device)


    # Training loop
    for epoch in range(n_epochs):
            
        if hill_climb == False:
   
            criterion = torch.nn.MSELoss() #criterion is actually equal to our funky multiobjective term here I think
            criterion.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            train_loss = mlp_backprop.train(model, train_loader, criterion, optimizer, device)

            print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")
            training_history["train_loss"].append(train_loss)


        else:
            criterion = torch.nn.MSELoss() 
            criterion.to(device)
            hillclimb.hill_climb(model, train_loader,criterion, device,rng)

            hidden_loss = 0.0
            output_loss =0.0

            for inputs, labels, _ in train_loader:

                inputs, labels = inputs.to(device), labels.to(device)
   
                #take fully connected hidden layer output
                x_hat = model.reconstruct_layer(inputs)
                #accumulate the loss
                hidden_loss += criterion(x_hat*2-1, inputs).item()
                
                #take the task output
                y = model.regress(inputs)
                output_loss+= criterion(y, labels).item()


            #len of train loader is one here (but we may want batches with other data)
            hidden_loss /= len(train_loader) 

            output_loss /= len(train_loader)


            print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {hidden_loss:.4f}")
            training_history["encoder_train_loss"].append(hidden_loss)

            
            print(f"Epoch [{epoch + 1}/{n_epochs}], Regression Training Loss: {output_loss:.4f}")
            training_history["task_train_loss"].append(output_loss)   

        


    # Save metrics for the fold if requested
    if fold_id is not None and metrics_dict is not None:
        metrics_dict[f"{fold_id} {batch_size} {learning_rate}"] = {
            "train_loss": training_history["train_loss"][-n_epochs:], 
            "val_loss": training_history["val_loss"][-n_epochs:],
        }


    return model, training_history

   
