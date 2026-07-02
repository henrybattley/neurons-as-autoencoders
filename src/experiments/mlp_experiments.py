import os
import numpy as np
import torch, torchvision



from src.models.mlp import MLP
from src.datasets.NK import NKLandscape
from src.optimizers import mlp_backprop


#dict to store the training progress of the model 
global training_history
training_history = {
    "train_loss": [],
    "val_loss": [],
  }

def train_NK_model(NK_data, 
                        n_epochs=100, 
                        batch_size=64,
                        learning_rate=0.001,
                        USE_GPU=False,
                        patience=3,
                        fold_id=None,
                        metrics_dict=None,
                        final_train=True):

    #variables to control early stopping (hoping to prevent overfitting)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Optionally use GPU if available (students GPU is not cuda enabled so skipped this by defaulting GPU use to false)
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    #if not final training (make validation splits)
    if final_train== False:
        # Create train-val split
        train_size = int(0.8 * len(NK_data))
        val_size = len(NK_data) - train_size

        train_data, val_data = torch.utils.data.random_split(NK_data, [train_size, val_size])

        val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             )
    #otherwise we use the whole training dataset
    else:
        train_data = NK_data
        val_loader=None


    train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             )
    

    # Initialize model, loss function, and optimizer (mainly left as boiler plate (Adam is state of the art optimiser and cross entropy loss is widely used))
    
    sample_x, _ = train_data[0]
    input_dim = sample_x.shape[0]

    print(f"input dims are {input_dim} should be 3")
    model = MLP(input_dim=input_dim,hidden_dim=2)
    model.to(device)
    criterion = torch.nn.MSELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):
        train_loss = mlp_backprop.train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")

        #if we are not in final training..
        if val_loader is not None:
            val_loss= mlp_backprop.eval(model, val_loader, criterion, device)
            training_history["val_loss"].append(val_loss)

            print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {val_loss:.4f}")

            #early stop when we have no validation loss improvement for three consecutive epochs
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        #otherwise we are in final training and disregard validation sets
        else:
            val_loss=None
         
    
        training_history["train_loss"].append(train_loss)

    # Save metrics for the fold if requested
    if fold_id is not None and metrics_dict is not None:
        metrics_dict[f"{fold_id} {batch_size} {learning_rate}"] = {
            "train_loss": training_history["train_loss"][-n_epochs:], 
            "val_loss": training_history["val_loss"][-n_epochs:],
        }
    
    # Return the model's state_dict (weights) - DO NOT CHANGE THIS
    return model.state_dict()




def main():
    # example usage
    # you could create a separate file that calls train_fashion_model with different parameters
    # or modify this as needed to add cross-validation, hyperparameter tuning, etc.
    
    ### ^^^ Seperate file created as suggested that calls train_fashion_model... please refer to visualise_dataset.ipynb
    
    NK_data = NKLandscape.get_dataset(N=3,K=1,n_samples=5000,seed=42)

    model_weights = train_NK_model(NK_data)

    # Save model weights

    #model_save_path = os.path.join('submission', 'model_weights.pth')
    #torch.save(model_weights, f=model_save_path)


if __name__ == "__main__":
    main()