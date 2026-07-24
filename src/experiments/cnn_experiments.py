import os
import numpy as np
import torch, torchvision
import random

from src.models import cnn_model
from src.models import cnn_model_no_pooling


from src.optimizers import global_backprop

import time


def train_cnn(  data, 
                input_dims,
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                n_filters=16,
                stride=1,
                padding=1,
                kernel_size=3,
                pool_kernel_size=2,
                pool_stride=2,
                n_classes=10,
                bias=True,
                seed=42):
    

    training_history = {
    "train_loss": [],
    "train_accuracy":[],
    "test_loss": []
    }
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is: {device}")

    #seed randomness 
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    g = torch.Generator()
    g.manual_seed(seed)

    #starting time from data loading
    start = time.perf_counter()

    train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    num_workers=0,  
    pin_memory=True,
    )


    model = cnn_model.CNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        classes=n_classes,
        bias=bias
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop..
    for epoch in range(n_epochs):
            
        train_loss,accuracy = global_backprop.train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.2f}")

        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history,elapsed



def train_no_pool_cnn(  data, 
                input_dims,
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                n_filters=16,
                stride=1,
                padding=1,
                kernel_size=3,
                pool_kernel_size=2,
                pool_stride=2,
                n_classes=10,
                bias=True,
                seed=42):
    

    training_history = {
    "train_loss": [],
    "train_accuracy":[],
    "test_loss": []
    }
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is: {device}")

    #seed randomness 
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    g = torch.Generator()
    g.manual_seed(seed)

    #starting time from data loading
    start = time.perf_counter()

    train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    num_workers=0,  
    pin_memory=True,
    )


    model = cnn_model_no_pooling.CNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        classes=n_classes,
        bias=bias
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop..
    for epoch in range(n_epochs):
            
        train_loss,accuracy = global_backprop.train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.2f}")

        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history,elapsed




""" deeper model to come back to 
def train_deep_cnn(  data, 
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

    #seed randomness 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    
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

    model = cnn_model.DeeperCNN(
    depth=10,
    n_filters=16,
    num_classes=10
    )
    model.to(device)

    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop

    for epoch in range(n_epochs):
            
        train_loss,accuracy = global_backprop.train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.4f}")

        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history,elapsed
"""


