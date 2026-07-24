import os
import numpy as np
import torch, torchvision
import random
import time

from src.models import nan_cnn
from src.models import weight_share_nan_cnn
from src.models import simple_weight_share_nan_cnn




def train_nan_cnn(  data, 
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

    #defining the FilterCNN model (network of filter autoencoders with classifier head)
    model = nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        classes=n_classes
    ).to(device)
    

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=learning_rate
        )
        for j in range(n_filters)
    ]

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

            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):
                
                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(images, j)

                loss = encoder_criterion(x_hat, images)

                loss.backward()

                optimizer.step()

                encoder_epoch_loss += loss.item()
            
            #after filters have updated as per their gradient info, 
            # perform individual forward passes through the filters, concatenate and extract resultant feature maps
            #with torch.no_grad():   #be sure not to compute gradients of forward passes
                #features = model.extract_features(images)

            #features = features.detach() 


            classifier_optimizer.zero_grad()

            #logits =model(images)

            #logits = model.classify(features)

            


            with torch.no_grad():
                features = model.extract_features(images)

            logits = model.classify(features)
            
            loss = classifier_criterion(logits, labels)

            loss.backward()

            classifier_optimizer.step()

            classifier_epoch_loss += loss.item()

            # Classification accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        
        #for average autoencoder loss, divide by the batch size and then the n filters
        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        avg_encoder_loss /= n_filters

        #for average classification loss, divide by the batch size and then form as percentage
        avg_classifier_loss = classifier_epoch_loss / len(train_loader)
        classification_accuracy = 100.0 * correct / total


        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {avg_encoder_loss:.4f}")

        training_history["encoder_train_loss"].append(avg_encoder_loss)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {avg_classifier_loss:.4f}, Accuracy: {classification_accuracy:.2f}%")

        training_history["task_train_loss"].append(avg_classifier_loss)
        
        training_history["train_accuracy"].append(classification_accuracy)
        
        
    elapsed = time.perf_counter() - start

    return model, training_history, elapsed



def train_weight_share_nan_cnn(  data, 
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

    #defining the FilterCNN model (network of filter autoencoders with classifier head)
    model = weight_share_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        classes=n_classes
    ).to(device)
    

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=learning_rate
        )
        for j in range(n_filters)
    ]

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

            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):
                
                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(images, j)

                loss = encoder_criterion(x_hat, images)

                loss.backward()

                optimizer.step()

                encoder_epoch_loss += loss.item()
            
            #after filters have updated as per their gradient info, 
            # perform individual forward passes through the filters, concatenate and extract resultant feature maps
            #with torch.no_grad():   #be sure not to compute gradients of forward passes
                #features = model.extract_features(images)

            #features = features.detach() 


            classifier_optimizer.zero_grad()

            #logits =model(images)

            #logits = model.classify(features)

            


            with torch.no_grad():
                features = model.extract_features(images)

            logits = model.classify(features)
            
            loss = classifier_criterion(logits, labels)

            loss.backward()

            classifier_optimizer.step()

            classifier_epoch_loss += loss.item()

            # Classification accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        
        #for average autoencoder loss, divide by the batch size and then the n filters
        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        avg_encoder_loss /= n_filters

        #for average classification loss, divide by the batch size and then form as percentage
        avg_classifier_loss = classifier_epoch_loss / len(train_loader)
        classification_accuracy = 100.0 * correct / total


        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {avg_encoder_loss:.4f}")

        training_history["encoder_train_loss"].append(avg_encoder_loss)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {avg_classifier_loss:.4f}, Accuracy: {classification_accuracy:.2f}%")

        training_history["task_train_loss"].append(avg_classifier_loss)
        
        training_history["train_accuracy"].append(classification_accuracy)
        
        
    elapsed = time.perf_counter() - start

    return model, training_history, elapsed


""" ---------------------------------------------------------------
-----------------------------------------------------------------"""
def train_linear_schedule_weight_share_nan_cnn(  data, 
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
                    seed=42,
                    patience=3):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "train_accuracy": [],
    "epoch_converged": []
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

    #defining the FilterCNN model (network of filter autoencoders with classifier head)
    model = weight_share_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        classes=n_classes
    ).to(device)
    

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=learning_rate
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=learning_rate)

    #initially setting best loss to be inf (used for early stopping logic)
    best_loss = float('inf')
    epochs_no_improve = 0
    min_delta = 1e-4


    for epoch in range(n_epochs):

        encoder_epoch_loss =0.0

        #per batch
        for images, _ in train_loader:

            images = images.to(device)
            
            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):
                
                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(images, j)

                loss = encoder_criterion(x_hat, images)

                loss.backward()

                optimizer.step()

                encoder_epoch_loss += loss.item()
            

        #for average autoencoder loss, divide by the batch size and then the n filters
        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        avg_encoder_loss /= n_filters

        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {avg_encoder_loss:.4f}")

        training_history["encoder_train_loss"].append(avg_encoder_loss)

        #early stop when we have no loss improvement for three consecutive epochs

        if avg_encoder_loss < best_loss - min_delta:
                best_loss = avg_encoder_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
        else:
                epochs_no_improve += 1

                if epochs_no_improve >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}... Now training classifier")
                        training_history["epoch_converged"].append(best_epoch)
                        break #stop training epochs

    #now train classifier

    for epoch in range(n_epochs):

        classifier_epoch_loss =0.0

        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            #extract features for the batch
            with torch.no_grad():
                features = model.extract_features(images)




            classifier_optimizer.zero_grad()

            logits = model.classify(features)
            
            loss = classifier_criterion(logits, labels)

            loss.backward()

            classifier_optimizer.step()

            classifier_epoch_loss += loss.item()

            # Classification accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        

        #for average classification loss, divide by the batch size and then form as percentage
        avg_classifier_loss = classifier_epoch_loss / len(train_loader)
        classification_accuracy = 100.0 * correct / total


        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {avg_classifier_loss:.4f}, Accuracy: {classification_accuracy:.2f}%")

        training_history["task_train_loss"].append(avg_classifier_loss)
        
        training_history["train_accuracy"].append(classification_accuracy)
        
        
    elapsed = time.perf_counter() - start

    return model, training_history, elapsed



""" ---------------------------------------------------------------
-----------------------------------------------------------------"""
def train_linear_schedule_simple_weight_share_nan_cnn(  data, 
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
                    output_padding=0,
                    n_classes=10,
                    seed=42,
                    patience=3):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "train_accuracy": [],
    "epoch_converged": []
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

    #defining the FilterCNN model (network of filter autoencoders with classifier head)
    model = simple_weight_share_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        output_padding=output_padding,
        classes=n_classes
    ).to(device)
    

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=learning_rate
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=learning_rate)

    #initially setting best loss to be inf (used for early stopping logic)
    best_loss = float('inf')
    epochs_no_improve = 0
    min_delta = 1e-4


    for epoch in range(n_epochs):

        encoder_epoch_loss =0.0

        #per batch
        for images, _ in train_loader:

            images = images.to(device)
            
            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):
                
                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(images, j)

                loss = encoder_criterion(x_hat, images)

                loss.backward()

                optimizer.step()

                encoder_epoch_loss += loss.item()
            

        #for average autoencoder loss, divide by the batch size and then the n filters
        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        avg_encoder_loss /= n_filters

        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {avg_encoder_loss:.4f}")

        training_history["encoder_train_loss"].append(avg_encoder_loss)

        #early stop when we have no loss improvement for three consecutive epochs

        if avg_encoder_loss < best_loss - min_delta:
                best_loss = avg_encoder_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
        else:
                epochs_no_improve += 1

                if epochs_no_improve >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}... Now training classifier")
                        training_history["epoch_converged"].append(best_epoch)
                        break #stop training epochs

    #now train classifier

    for epoch in range(n_epochs):

        classifier_epoch_loss =0.0

        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            #extract features for the batch
            with torch.no_grad():
                features = model.extract_features(images)




            classifier_optimizer.zero_grad()

            logits = model.classify(features)
            
            loss = classifier_criterion(logits, labels)

            loss.backward()

            classifier_optimizer.step()

            classifier_epoch_loss += loss.item()

            # Classification accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        

        #for average classification loss, divide by the batch size and then form as percentage
        avg_classifier_loss = classifier_epoch_loss / len(train_loader)
        classification_accuracy = 100.0 * correct / total


        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {avg_classifier_loss:.4f}, Accuracy: {classification_accuracy:.2f}%")

        training_history["task_train_loss"].append(avg_classifier_loss)
        
        training_history["train_accuracy"].append(classification_accuracy)
        
        
    elapsed = time.perf_counter() - start

    return model, training_history, elapsed



def train_linear_schedule_no_pool_simple_weight_share_nan_cnn(  data, 
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
                    output_padding=0,
                    n_classes=10,
                    seed=42,
                    patience=3):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "train_accuracy": [],
    "epoch_converged": []
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

    #defining the FilterCNN model (network of filter autoencoders with classifier head)
    model = simple_weight_share_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        output_padding=output_padding,
        classes=n_classes
    ).to(device)
    

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=learning_rate
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=learning_rate)

    #initially setting best loss to be inf (used for early stopping logic)
    best_loss = float('inf')
    epochs_no_improve = 0
    min_delta = 1e-4


    for epoch in range(n_epochs):

        encoder_epoch_loss =0.0

        #per batch
        for images, _ in train_loader:

            images = images.to(device)
            
            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):
                
                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(images, j)

                loss = encoder_criterion(x_hat, images)

                loss.backward()

                optimizer.step()

                encoder_epoch_loss += loss.item()
            

        #for average autoencoder loss, divide by the batch size and then the n filters
        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        avg_encoder_loss /= n_filters

        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {avg_encoder_loss:.4f}")

        training_history["encoder_train_loss"].append(avg_encoder_loss)

        #early stop when we have no loss improvement for three consecutive epochs

        if avg_encoder_loss < best_loss - min_delta:
                best_loss = avg_encoder_loss
                best_epoch = epoch + 1
                epochs_no_improve = 0
        else:
                epochs_no_improve += 1

                if epochs_no_improve >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}... Now training classifier")
                        training_history["epoch_converged"].append(best_epoch)
                        break #stop training epochs

    #now train classifier

    for epoch in range(n_epochs):

        classifier_epoch_loss =0.0

        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            #extract features for the batch
            with torch.no_grad():
                features = model.extract_features(images)




            classifier_optimizer.zero_grad()

            logits = model.classify(features)
            
            loss = classifier_criterion(logits, labels)

            loss.backward()

            classifier_optimizer.step()

            classifier_epoch_loss += loss.item()

            # Classification accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        

        #for average classification loss, divide by the batch size and then form as percentage
        avg_classifier_loss = classifier_epoch_loss / len(train_loader)
        classification_accuracy = 100.0 * correct / total


        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {avg_classifier_loss:.4f}, Accuracy: {classification_accuracy:.2f}%")

        training_history["task_train_loss"].append(avg_classifier_loss)
        
        training_history["train_accuracy"].append(classification_accuracy)
        
        
    elapsed = time.perf_counter() - start

    return model, training_history, elapsed








""" training loop that trains filters by running entirely on train data before moving to the next filter then classifier, then repeat
    # Training loop
    for epoch in range(n_epochs):

        encoder_loss=0.0

        for filter in range(n_filters):

            train_loss = nan_cnn_local_gd.train_filters(model, train_loader, encoder_criterion,
                                                        filter_optimizers[filter], device, filter)

            encoder_loss += train_loss

        #divide by how many neurons in the hidden layer (so we average over the hidden neurons)
        encoder_loss /= n_filters
        
        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(encoder_loss)

        epoch_loss = 0.0
    
        train_loss = nan_cnn_local_gd.train_classifier(model, train_loader, classifer_criterion, classifier_optimizer, device)


        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {train_loss:.4f}")
        training_history["task_train_loss"].append(train_loss)
    """ 