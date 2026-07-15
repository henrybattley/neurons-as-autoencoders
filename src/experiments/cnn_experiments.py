import os
import numpy as np
import torch, torchvision
import random

from src.models import cnn_model
from src.optimizers import global_backprop


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
    

    #student's gpu is non-CUDA enabled
    device = torch.device('cuda')

    #seed randomness (already performed in notebook but now training is self-contained)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_loader = torch.utils.data.DataLoader( data,
                                                batch_size=batch_size,
                                                shuffle=True
                                                )

    #sample and retrieve shape for parameterizing model input dims
    sample_x, _= data[0]
    input_dim = sample_x.shape[0]

    print(f"input dims are {input_dim}")


    model = cnn_model.CNN()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):
            
        train_loss = global_backprop.train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")

        training_history["train_loss"].append(train_loss)



    return model, training_history