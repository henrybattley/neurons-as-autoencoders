import os
import numpy as np
import torch, torchvision
import random

from src.models import cnn_model
from src.models import nan_cnn

from src.optimizers import global_backprop
from src.optimizers import nan_cnn_local_gd

import time


#Training pipeline, when hill_climb=True training follows that as defined by Bull 
def train_cnn(  data, 
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                seed=42):
    
    training_history = {
    "train_loss": [],
    "train_accuracy":[],
    "test_loss": []
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



    #starting timing from where the models differ
    start = time.perf_counter()

    model = cnn_model.CNN()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop

    for epoch in range(n_epochs):
            
        train_loss,accuracy = global_backprop.train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.2f}")

        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history,elapsed


