import os
import numpy as np
import torch, torchvision
import random


from src.models.mlp import MLP
from src.datasets.NK import NKLandscape
from src.optimizers import global_backprop
from src.optimizers import MLP_hillclimb


#dict to store the training progress of the model 
#global training_history
""" 
training_history = {
    "train_loss": [],
    "test_loss": [],
  }"""

#training pipeline with default parameters
def train_MLP(  NK_data_train, 
                NK_data_test,
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                USE_GPU=False,
                fold_id=None,
                metrics_dict=None,
                final_train=True,
                hill_climb=True,
                seed=42):
    
    training_history = {
    "train_loss": [],
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
    test_loader = torch.utils.data.DataLoader(  NK_data_test,
                                                batch_size=batch_size,
                                                shuffle=False if hill_climb else True, 
                                                )
    

    #sample and retrieve shape for parameterizing model input dims
    sample_x, _ = NK_data_train[0]
    input_dim = sample_x.shape[0]

    print(f"input dims are {input_dim}")


    model = MLP(input_dim=input_dim,hidden_dim=10) # 10 nodes in the hidden layer (H of 10) as per Bull's experiments

    model.to(device)
    criterion = torch.nn.MSELoss()
    criterion.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):
            
        if hill_climb == False:
            train_loss = global_backprop.train(model, train_loader, criterion, optimizer, device)

        else:
            train_loss = MLP_hillclimb.mlp_hill_climb(model, train_loader, criterion, device,rng)


        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")
            
        
        training_history["train_loss"].append(train_loss)

        # compute the test loss every epoch
        #if epoch % 1000 == 0 or epoch==9999:

        test_epoch_loss = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_epoch_loss += loss.item()

        avg_loss = test_epoch_loss / len(test_loader)
        training_history['test_loss'].append(avg_loss)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Test Loss: {avg_loss:.4f}")



    # Save metrics for the fold if requested
    if fold_id is not None and metrics_dict is not None:
        metrics_dict[f"{fold_id} {batch_size} {learning_rate}"] = {
            "train_loss": training_history["train_loss"][-n_epochs:], 
            "val_loss": training_history["val_loss"][-n_epochs:],
        }


    return model, training_history

   




# main is not really needed if we just run experiments from jupyter 
#def main():

    
    #NK_data = NKLandscape.get_dataset(N=20,K=10,n_samples=5000,seed=42)

    #model_weights = train_MLP(NK_data)

    # Save model weights

    #model_save_path = os.path.join('submission', 'model_weights.pth')
    #torch.save(model_weights, f=model_save_path)


#if __name__ == "__main__":
    #main()