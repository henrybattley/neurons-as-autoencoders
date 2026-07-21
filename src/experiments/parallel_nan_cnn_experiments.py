import os
import numpy as np
import torch, torchvision
import random
import time

from src.models import parallel_nan_cnn

torch.backends.cudnn.benchmark = True


def train_parallel_nan_cnn(  data, 
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
                    seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "train_accuracy": []
    }
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is: {device}")

    #seed randomness (already performed in notebook but now training is self-contained)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

        
    #starting time from data loading
    start = time.perf_counter()

    train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  
    pin_memory=True,
    )

    #defining the FilterCNN model (network of filter autoencoders with classifier head)
    """ 
    model = parallel_nan_cnn.GroupedLocalAutoencoders(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        classes=n_classes
    ).to(device)"""

    model = parallel_nan_cnn.GroupedLocalAutoencoders(kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding,
                                                      n_filters=n_filters,
                                                      classes=n_classes).to(device)

    
    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    """ 
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=learning_rate
        )
        for j in range(n_filters)
    ]"""

    # 1. Autoencoder optimizer updates grouped encoder and decoder weights simultaneously
    autoencoder_optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=learning_rate
    )


    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=learning_rate)


    for epoch in range(n_epochs):

        encoder_epoch_loss =0.0
        classifier_epoch_loss =0.0

        correct = 0
        total = 0

        #per batch
        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # --- 1. PARALLEL AUTOENCODER UPDATE ---
            autoencoder_optimizer.zero_grad()

            # Reconstruct all N filters in 1 parallel GPU pass
            x_hat, x_expanded, _ = model.reconstruct_all(images)

            # Compute parallel MSE loss across all N isolated filter maps
            ae_loss = encoder_criterion(x_hat, x_expanded)

            #backward pass is split into n_filters independent gradient calculations
            ae_loss.backward()

            autoencoder_optimizer.step()

            # Track total local encoder loss
            encoder_epoch_loss += ae_loss.detach()

        
            classifier_optimizer.zero_grad()


            with torch.no_grad():
                features = model.extract_features(images)

            logits = model.classify(features)
            
            loss = classifier_criterion(logits, labels)

            loss.backward()

            classifier_optimizer.step()

            classifier_epoch_loss += loss.detach()

            # Classification accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum()
            total += labels.size(0)

        
        #for average autoencoder loss, divide by the batch size and then the n filters
        avg_encoder_loss = (encoder_epoch_loss.item() / len(train_loader))

        #for average classification loss, divide by the batch size and then form as percentage
        avg_classifier_loss = classifier_epoch_loss.item() / len(train_loader)
        classification_accuracy = 100.0 * correct.item() / total


        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {avg_encoder_loss:.4f}")

        training_history["encoder_train_loss"].append(avg_encoder_loss)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {avg_classifier_loss:.4f}, Accuracy: {classification_accuracy:.4f}%")

        training_history["task_train_loss"].append(avg_classifier_loss)
        
        training_history["train_accuracy"].append(classification_accuracy)
        
        
    elapsed = time.perf_counter() - start

    return model, training_history, elapsed
