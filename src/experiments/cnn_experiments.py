import os
import numpy as np
import torch, torchvision
import random

from src.models import cnn_model
from src.models import cnn_model_no_pooling
from src.models import conv_autoencoder



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


def train_cnn_get_features(  data, 
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
                epochs_to_show=[1],
                seed=42):
    

    training_history = {
    "train_loss": [],
    "train_accuracy":[],
    "test_loss": []
    }

    feature_history= {}
    

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

    #image for inspection..
    probe_image, probe_label = data[0]
    probe_image = probe_image.unsqueeze(0).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop..
    for epoch in range(n_epochs):


        if (epoch) in epochs_to_show:

            with torch.no_grad():

                #get the unpooled feature maps
                maps = model.feature_maps(probe_image)

                pooled_maps = model.pool(maps)

                cnn_weights = model.conv1.weight.detach().cpu().clone()
                
                logits = model(probe_image)
                prediction = logits.argmax(1).item()

                feature_history[epoch] = {
                    "label": probe_label,
                    "prediction":prediction,
                    "logits": logits.cpu().clone(),
                    "original": probe_image.cpu().clone(),
                    "maps": maps.cpu().clone(),
                    "pooled_maps": pooled_maps.cpu().clone(),
                    "kernel_weights": cnn_weights.cpu(),
                }


            
        train_loss,accuracy = global_backprop.train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.2f}")

        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(accuracy)




    elapsed = time.perf_counter() - start
    return model, training_history, feature_history, elapsed



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
    "task_train_loss": [],
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



def train_ae_cnn_get_features(  data, 
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
                epochs_to_show=[1],
                seed=42):
    

    training_history = {
    "train_loss": [],
    "train_accuracy":[],
    "encoder_train_loss": [],


    }

    feature_history= {}
    

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


    model = conv_autoencoder.CNN_AE(
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

    #image for inspection..
    probe_image, probe_label = data[0]
    probe_image = probe_image.unsqueeze(0).to(device)


    encoder_criterion = torch.nn.MSELoss()
    encoder_criterion.to(device)

    task_criterion = torch.nn.CrossEntropyLoss()
    task_criterion.to(device)


    ae_optimizer = torch.optim.Adam(
    list(model.encoder.parameters()) +
    list(model.decoder.parameters()),
    lr=learning_rate
)

    classifier_optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)


    # Training loop..
    for epoch in range(n_epochs):


        if (epoch) in epochs_to_show:

            with torch.no_grad():

                #get the unpooled feature maps
                maps = model.encode(probe_image)

                pooled_maps = model.pool(maps)

                encoder_weights = model.encoder.weight.detach().cpu().clone()
                decoder_weights = model.decoder.weight.detach().cpu().clone()

                
                logits = model.classify(maps)

                prediction = logits.argmax(1).item()

                feature_history[epoch] = {
                    "label": probe_label,
                    "prediction":prediction,
                    "logits": logits.cpu().clone(),
                    "original": probe_image.cpu().clone(),
                    "maps": maps.cpu().clone(),
                    "pooled_maps": pooled_maps.cpu().clone(),
                    "encoder_weights": encoder_weights.cpu(),
                    "decoder_weights": decoder_weights.cpu()
                }


            
        #train_loss,accuracy = global_backprop.train(model, train_loader, criterion, optimizer, device)

        model.train()

        encoder_epoch_loss=0.0
        task_epoch_loss = 0.0

        correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            ae_optimizer.zero_grad()

            x_hat = model.autoencode(inputs)
            encoder_loss = encoder_criterion(x_hat,inputs)

            encoder_loss.backward()
            ae_optimizer.step()

            encoder_epoch_loss+=encoder_loss

            #now train the classifier

            classifier_optimizer.zero_grad()

            features = model.encode(inputs)

            task_outputs = model.classify(features)

            task_loss = task_criterion(task_outputs, labels)
            task_loss.backward()
            classifier_optimizer.step()

            task_epoch_loss += task_loss.detach() 

            _, predicted = torch.max(task_outputs, 1)
            correct += (predicted == labels).sum()
            total_samples += labels.size(0)

        avg_encoder_loss = encoder_epoch_loss.item() / len(train_loader)
        #scale encoder loss to be the same as our filters as autoencoders method
        avg_encoder_loss /= n_filters

        avg_task_loss = task_epoch_loss.item() / len(train_loader)
        accuracy = correct.item() / total_samples


        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Loss: {avg_encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(avg_encoder_loss)


        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {avg_task_loss:.4f}, Training Accuracy: {accuracy:.2f}")
        training_history["task_train_loss"].append(avg_task_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history, feature_history, elapsed



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


