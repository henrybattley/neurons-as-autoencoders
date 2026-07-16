import os
import numpy as np
import torch, torchvision
import random

from src.models import cnn_model
from src.models import nan_cnn

from src.optimizers import global_backprop
from src.optimizers import nan_cnn_local_gd


#Training pipeline, when hill_climb=True training follows that as defined by Bull 
def train_cnn(  data, 
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                seed=42):
    
    training_history = {
    "train_loss": [],
    "test_loss": [],
    }
    

    #device = torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is: {device}")

    #seed randomness (already performed in notebook but now training is self-contained)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    """ 
    train_loader = torch.utils.data.DataLoader( data,
                                                batch_size=batch_size,
                                                shuffle=True
                                                )"""
    
    train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,  
    pin_memory=True,
    persistent_workers=True,
)

    #sample and retrieve shape for parameterizing model input dims
    sample_x, _= data[0]
    input_dim = sample_x.shape[0]

    print(f"input dims are {input_dim}")


    model = cnn_model.CNN()
    model.to(device)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):
            
        train_loss = global_backprop.train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")

        training_history["train_loss"].append(train_loss)



    return model, training_history



#Training pipeline, when hill_climb=True training follows that as defined by Bull 
def train_nan_cnn(  data, 
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                n_filters=16,
                seed=42):
    
    training_history = {
    "train_loss": [],
    "test_loss": [],
    }
    

    #device = torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is: {device}")

    #seed randomness (already performed in notebook but now training is self-contained)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    """ 
    train_loader = torch.utils.data.DataLoader( data,
                                                batch_size=batch_size,
                                                shuffle=True
                                                )"""
    
    train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,  
    pin_memory=True,
    persistent_workers=True,
)
    """ this is apparently it:
    
    optimizer = Adam(model.filters[j].parameters(), lr=...)

x_hat = model.reconstruct_filter(images, j)

loss = mse(x_hat, images)

loss.backward()

optimizer.step()


again this:

optimizer = torch.optim.Adam(
    model.filters[7].parameters(),
    lr=...
)

"""


    model = nan_cnn.FilterCNN() 
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    filter_optimizers = []

    for filter in range(n_filters):

        filter_optimizers.append(

            torch.optim.Adam(
            [
                model.encoder[filter],
                model.decoder[filter],
            ],
            lr=learning_rate)

        )
    
    #not sure here need to refactor
    output_optimizer = torch.optim.Adam(model.output.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):

        encoder_loss=0.0

        for filter in range(n_filters):

            train_loss = nan_cnn_local_gd.train_filters(model, train_loader, criterion, filter_optimizers[filter], device, filter)

            encoder_loss += train_loss

        #divide by how many neurons in the hidden layer (so we average over the hidden neurons)
        encoder_loss /= n_filters
        
        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(encoder_loss)

        epoch_loss = 0.0
    
        train_loss = nan_cnn_local_gd.train_classifier(model, train_loader, criterion, output_optimizer, device)


        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {train_loss:.4f}")
        training_history["task_train_loss"].append(train_loss)



    return model, training_history