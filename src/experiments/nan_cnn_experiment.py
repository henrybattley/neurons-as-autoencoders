import os
import numpy as np
import torch, torchvision
import random
import time

from src.models import nan_cnn
from src.models import weight_share_nan_cnn
#from src.models import simple_weight_share_nan_cnn
#from src.models import no_pool_simple_weight_share_nan_cnn
from src.models import no_pool_weight_share_nan_cnn
from src.models import no_pool_nan_cnn

from src.models import crelu_nan_cnn

from src.models import noisy_nan_cnn
from src.models import sparse_nan_cnn

from torchvision.transforms import functional as TF
import torch.nn.functional as F 






def train_nan_cnn(  data, 
                    input_dims,
                    n_epochs=100, 
                    batch_size=64,
                    dual_lr = False,
                    learning_rate=0.001, 
                    classifier_lr=0.0001,
                    ae_lr=0.001,
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
        bias=bias,
        classes=n_classes
    ).to(device)
    

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)

    #when using one singular learning rate for the optimisers
    if dual_lr == False:
         ae_lr = learning_rate
         classifier_lr = learning_rate
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=ae_lr
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=classifier_lr)


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




def train_nan_cnn_show_features(  data, 
                    input_dims,
                    n_epochs=100, 
                    batch_size=64,
                    dual_lr = False,
                    learning_rate=0.001, 
                    classifier_lr=0.0001,
                    ae_lr=0.001,
                    n_filters=16,
                    stride=1,
                    padding=1,
                    kernel_size=3,
                    pool_kernel_size=2,
                    pool_stride=2,
                    n_classes=10,
                    epochs_to_show=[1],
                    seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "train_accuracy": []
    }

    feature_history = {}
        

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

    #image to track across the epochs
    probe_image, probe_label = data[0]
    probe_image = probe_image.unsqueeze(0).to(device)
        

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)

    #when using one singular learning rate for the optimisers
    if dual_lr == False:
         ae_lr = learning_rate
         classifier_lr = learning_rate
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=ae_lr
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=classifier_lr)


    for epoch in range(n_epochs):

        if (epoch) in epochs_to_show:

            with torch.no_grad():

                #get the unpooled feature maps
                maps = model.feature_maps(probe_image)

                #get the pooled feature maps before flattening
                pooled_maps = model.pool(maps)

                #get the reconstruction for every filter
                recons = torch.cat(
                    [f(probe_image) for f in model.filters],
                    dim=1
                )

                #encoder weights
                weights = torch.stack([
                    f.encoder.weight.squeeze().cpu().clone()
                    for f in model.filters
                ])

                #decoder weights 
                decoder_weights = torch.stack([
                f.decoder.weight.squeeze().cpu().clone()
                for f in model.filters
            ])
                #make the prediction
                logits = model(probe_image)
                prediction = logits.argmax(1).item()

                feature_history[epoch] = {
                    "label": probe_label,
                    "prediction":prediction,
                    "logits": logits.cpu().clone(),
                    "original": probe_image.cpu().clone(),
                    "maps": maps.cpu().clone(),
                    "pooled_maps": pooled_maps.cpu().clone(),
                    "reconstructions": recons.cpu().clone(),
                    "encoder_weights": weights.cpu(),
                    "decoder_weights": decoder_weights.cpu()

                }

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
   

            classifier_optimizer.zero_grad()

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

    return model, training_history, feature_history, elapsed



def train_weight_share_nan_cnn(  data, 
                    input_dims,
                    in_channels=1,
                    n_epochs=100, 
                    dual_lr=False,
                    batch_size=64,
                    learning_rate=0.001,      
                    classifier_lr=0.001,
                    ae_lr=0.001,
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
        in_channels=in_channels
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        bias=bias,
        classes=n_classes
    ).to(device)
    

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)

    #when using one singular learning rate for the optimisers
    if dual_lr == False:
         ae_lr = learning_rate
         classifier_lr = learning_rate
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=ae_lr
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=classifier_lr)


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
                    bias=True,
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
        bias=bias,
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







def train_linear_schedule_no_pool_weight_share_nan_cnn(  data, 
                    input_dims,
                    n_epochs=100, 
                    batch_size=64,
                    learning_rate=0.001,
                    n_filters=16,
                    stride=1,
                    padding=1,
                    kernel_size=3,
                    output_padding=0,
                    n_classes=10,
                    bias=False,
                    sigmoid=True,
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
    model = no_pool_weight_share_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        output_padding=output_padding,
        bias=bias,
        sigmoid=sigmoid,
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

def train_parallel_schedule_no_pool_weight_share(  data, 
                    input_dims,
                    n_epochs=100, 
                    dual_lr=False,
                    batch_size=64,
                    learning_rate=0.001,      
                    classifier_lr=0.001,
                    ae_lr=0.001,
                    n_filters=16,
                    stride=1,
                    padding=1,
                    kernel_size=3,
                    output_padding=0,
                    n_classes=10,
                    bias=True,
                    sigmoid=True,
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
    model = no_pool_weight_share_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        output_padding=output_padding,
        sigmoid=sigmoid,
        bias=bias,
        classes=n_classes
    ).to(device)
    

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)

    #when using one singular learning rate for the optimisers
    if dual_lr == False:
         ae_lr = learning_rate
         classifier_lr = learning_rate
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=ae_lr
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=classifier_lr)


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



def train_parallel_schedule_no_pool_nan_cnn(data, 
                    input_dims,
                    n_epochs=100, 
                    batch_size=64,
                    dual_lr=False,
                    learning_rate=0.001,
                    classifier_lr=0.001,
                    ae_lr=0.00001,
                    n_filters=16,
                    stride=1,
                    padding=1,
                    kernel_size=3,
                    output_padding=0,
                    n_classes=10,
                    bias=True,
                    sigmoid=True,
                    seed=42,
                    ): 


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

    model = no_pool_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,

        output_padding=output_padding,
        bias=bias,
        sigmoid=sigmoid,
        classes=n_classes
    ).to(device)
    

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)

    #when using one singular learning rate for the optimisers
    if dual_lr == False:
         ae_lr = learning_rate
         classifier_lr = learning_rate
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=ae_lr
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=classifier_lr)


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
-----------------------------------------------------------------
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

        model = weight_share_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        bias=False,
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
"""


""" """

def train_linear_schedule_sep_weights_nan_cnn(  data, 
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
                    bias=True,
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
    model = nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        bias=bias,
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








def train_input_corruption_nan_cnn(  data, 
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
                    bias=False,
                    sigmoid=True,
                    sigma =0.1,
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
    model = no_pool_weight_share_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        output_padding=output_padding,
        bias=bias,
        sigmoid=sigmoid,
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

            # per batch, corrupt the input
            noise = sigma * torch.randn_like(images)
            corrupted = torch.clamp(images + noise, 0.0, 1.0)
            
            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):
                
                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()


                x_hat = model.reconstruct(corrupted,j)

                #x_hat = model.reconstruct(images, j)

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



def train_masked_nan_cnn(  data, 
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
                    bias=False,
                    sigmoid=True,
                    sigma =0.1,
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
    model = no_pool_weight_share_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        output_padding=output_padding,
        bias=bias,
        sigmoid=sigmoid,
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

            # per batch, corrupt the input

            mask = (torch.rand_like(images) > sigma).float()
            corrupted = images * mask

            
            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):
                
                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(corrupted,j)

                #x_hat = model.reconstruct(images, j)

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



def train_hidden_corruption_nan_cnn(  data, 
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
                    bias=False,
                    sigmoid=True,
                    latent_sigma=0.2,
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
    model = noisy_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        output_padding=output_padding,
        bias=bias,
        sigmoid=sigmoid,
        latent_sigma=latent_sigma,
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

            # per batch, corrupt the input

            #mask = (torch.rand_like(images) > sigma).float()
            #corrupted = images * mask

            
            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):
                
                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(images,j)

                #x_hat = model.reconstruct(images, j)

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



def train_sparse_nan_cnn(  data, 
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
                    bias=False,
                    sigmoid=True,
                    weight_decay=1e-2, 
                    seed=42,
                    lambda_sparse=1e-5,
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
    model = sparse_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        output_padding=output_padding,
        bias=bias,
        sigmoid=sigmoid,
        lambda_sparse=lambda_sparse,
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
            lr=learning_rate,
            weight_decay=weight_decay
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=learning_rate,weight_decay=weight_decay)

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

                x_hat,sparsity_loss = model.reconstruct(images, j)

                criterion_loss =  encoder_criterion(x_hat, images)

                #minimise the sparsity loss term with the criterion loss
                loss = sparsity_loss + criterion_loss

                loss.backward()

                optimizer.step()

                #just inspect criterion loss
                encoder_epoch_loss += criterion_loss.item()
            

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



def train_rotating_input_nan_cnn(  data, 
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
                    bias=False,
                    sigmoid=True,
                    weight_decay=1e-2, 
                    degree=5,
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
    model = no_pool_weight_share_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        output_padding=output_padding,
        bias=bias,
        sigmoid=sigmoid,
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
            lr=learning_rate,
            weight_decay=weight_decay
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=learning_rate,weight_decay=weight_decay)

    #initially setting best loss to be inf (used for early stopping logic)
    best_loss = float('inf')
    epochs_no_improve = 0
    min_delta = 1e-4


    for epoch in range(n_epochs):

        encoder_epoch_loss =0.0

        #per batch
        for images, _ in train_loader:

            images = images.to(device)

            #with 0.5 prob do the small rotation
            if torch.rand((), device=device) < 0.5:

                angles = torch.empty(images.size(0),device=device).uniform_(-degree, degree)

                rotated = torch.stack([
                TF.rotate(img, float(angle), fill=0)
                for img, angle in zip(images, angles)
            ])

            else: 
                 rotated= images
            
            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):
                
                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(rotated, j)

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




def train_nan_cnn_diverse_filters_show_features(  data, 
                    input_dims,
                    n_epochs=100, 
                    batch_size=64,
                    dual_lr = False,
                    learning_rate=0.001, 
                    classifier_lr=0.0001,
                    ae_lr=0.001,
                    n_filters=16,
                    stride=1,
                    padding=1,
                    kernel_size=3,
                    pool_kernel_size=2,
                    pool_stride=2,
                    n_classes=10,
                    epochs_to_show=[1],
                    lambda_frob=0.2,
                    seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "train_accuracy": []
    }

    feature_history = {}
        

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

    #image to track across the epochs
    probe_image, probe_label = data[0]
    probe_image = probe_image.unsqueeze(0).to(device)
        

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)

    #when using one singular learning rate for the optimisers
    if dual_lr == False:
         ae_lr = learning_rate
         classifier_lr = learning_rate
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=ae_lr
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=classifier_lr)


    for epoch in range(n_epochs):

        if (epoch) in epochs_to_show:

            with torch.no_grad():

                #get the unpooled feature maps
                maps = model.feature_maps(probe_image)

                #get the pooled feature maps before flattening
                pooled_maps = model.pool(maps)

                #get the reconstruction for every filter
                recons = torch.cat(
                    [f(probe_image) for f in model.filters],
                    dim=1
                )

                #encoder weights
                weights = torch.stack([
                    f.encoder.weight.squeeze().cpu().clone()
                    for f in model.filters
                ])

                #decoder weights 
                decoder_weights = torch.stack([
                f.decoder.weight.squeeze().cpu().clone()
                for f in model.filters
            ])
                #make the prediction
                logits = model(probe_image)
                prediction = logits.argmax(1).item()

                feature_history[epoch] = {
                    "label": probe_label,
                    "prediction":prediction,
                    "logits": logits.cpu().clone(),
                    "original": probe_image.cpu().clone(),
                    "maps": maps.cpu().clone(),
                    "pooled_maps": pooled_maps.cpu().clone(),
                    "reconstructions": recons.cpu().clone(),
                    "encoder_weights": weights.cpu(),
                    "decoder_weights": decoder_weights.cpu()

                }

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

                #get the frobenius here, updated with every filter 
                W = torch.stack([
                    f.encoder.weight.view(-1)
                    for f in model.filters
                ])
                #print(f"W shape is: {W.shape}")


                #W_frob = torch.linalg.matrix_norm(W)
                #W_frob = torch.sqrt(torch.trace(W.T@W))

                I = I = torch.eye(W.shape[1], device=device)
                # row or column formulation? (the below makes more sense since we get the 9x9 kernel matrix)
                orth = W.T@W - I

                #orth_frob = torch.sqrt(torch.trace(orth.T@orth))
                loss_frob = torch.linalg.matrix_norm(orth, ord='fro')**2

                
                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(images, j)

                loss = encoder_criterion(x_hat, images) + lambda_frob* loss_frob

                loss.backward()

                optimizer.step()

                encoder_epoch_loss += loss.item()
   

            classifier_optimizer.zero_grad()

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

    return model, training_history, feature_history, elapsed


""" I would say get the other weights only when not equal to the current filter being optimised
                weights = torch.stack([
                f.encoder.weight.view(-1).detach()
                for f in model.filters
                ]) """

#take the norm of all other weights
#weights = F.normalize(weights, dim=1)

#similarities = weights @ weights[j]

#loss_div = similarities.pow(2).sum() - similarities[j].pow(2)

# scale by how many filters contained within the similarity computation
#loss_div /= (n_filters - 1)




def train_nan_cnn_diverse_filters_show_features_localised(  data, 
                    input_dims,
                    n_epochs=100, 
                    batch_size=64,
                    dual_lr = False,
                    learning_rate=0.001, 
                    classifier_lr=0.0001,
                    ae_lr=0.001,
                    n_filters=16,
                    stride=1,
                    padding=1,
                    kernel_size=3,
                    pool_kernel_size=2,
                    pool_stride=2,
                    n_classes=10,
                    epochs_to_show=[1],
                    lambda_cosine=0.2,
                    seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "recon_train_loss": [],
    "sim_train_loss": [],
    "task_train_loss": [],
    "train_accuracy": []
    }

    feature_history = {}
        

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

    #image to track across the epochs
    probe_image, probe_label = data[0]
    probe_image = probe_image.unsqueeze(0).to(device)
        

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)

    #when using one singular learning rate for the optimisers
    if dual_lr == False:
         ae_lr = learning_rate
         classifier_lr = learning_rate
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=ae_lr
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=classifier_lr)


    for epoch in range(n_epochs):

        if (epoch) in epochs_to_show:

            with torch.no_grad():

                #get the unpooled feature maps
                maps = model.feature_maps(probe_image)

                #get the pooled feature maps before flattening
                pooled_maps = model.pool(maps)

                #get the reconstruction for every filter
                recons = torch.cat(
                    [f(probe_image) for f in model.filters],
                    dim=1
                )

                #encoder weights
                weights = torch.stack([
                    f.encoder.weight.squeeze().cpu().clone()
                    for f in model.filters
                ])

                #decoder weights 
                decoder_weights = torch.stack([
                f.decoder.weight.squeeze().cpu().clone()
                for f in model.filters
            ])
                #make the prediction
                logits = model(probe_image)
                prediction = logits.argmax(1).item()

                feature_history[epoch] = {
                    "label": probe_label,
                    "prediction":prediction,
                    "logits": logits.cpu().clone(),
                    "original": probe_image.cpu().clone(),
                    "maps": maps.cpu().clone(),
                    "pooled_maps": pooled_maps.cpu().clone(),
                    "reconstructions": recons.cpu().clone(),
                    "encoder_weights": weights.cpu(),
                    "decoder_weights": decoder_weights.cpu()

                }

        encoder_epoch_loss =0.0
        recon_epoch_loss=0.0
        similarity_epoch_loss=0.0
        classifier_epoch_loss =0.0

        correct = 0
        total = 0

        #per batch
        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)


            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):


                w_j = model.filters[j].encoder.weight.view(-1)
                loss_div = 0


                for k in range(n_filters):

                    if k == j:
                        continue
                    #Filter j observes the current state of its neighbours but does not try to update them
                    w_k = model.filters[k].encoder.weight.detach().view(-1)

                    loss_div += F.cosine_similarity(
                        w_j.unsqueeze(0),
                        w_k.unsqueeze(0)
                    ).pow(2)
                

                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(images, j)

                recon_loss = encoder_criterion(x_hat, images)
                similarity = lambda_cosine* loss_div

                loss = recon_loss + similarity

                #print(f"recon: {recon_loss} similarity (scaled): {similarity}")

                loss.backward()

                optimizer.step()

                encoder_epoch_loss += loss.item()
                recon_epoch_loss += recon_loss.item()
                similarity_epoch_loss += similarity.item()
   

            classifier_optimizer.zero_grad()

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

        #same for the distinct reconstruction error
        avg_recon_loss = recon_epoch_loss / len(train_loader)
        avg_recon_loss /= n_filters

        #same for the similarity between filters
        avg_similarity_loss = similarity_epoch_loss / len(train_loader)
        avg_similarity_loss /= n_filters


        #for average classification loss, divide by the batch size and then form as percentage
        avg_classifier_loss = classifier_epoch_loss / len(train_loader)
        classification_accuracy = 100.0 * correct / total

        #print and append whole loss term used for training the autoencoder
        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {avg_encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(avg_encoder_loss)

        #print and append the average reconstruction loss for the autoencoder
        print(f"Epoch [{epoch + 1}/{n_epochs}], Average reconstruction Loss: {avg_recon_loss:.4f}")
        training_history["recon_train_loss"].append(avg_recon_loss)

        #print and append the average cosine similarity between encoder filters
        print(f"Epoch [{epoch + 1}/{n_epochs}], Average similarity: {avg_similarity_loss:.4f}")
        training_history["sim_train_loss"].append(avg_similarity_loss)

        #print and append classification performance
        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {avg_classifier_loss:.4f}, Accuracy: {classification_accuracy:.2f}%")

        training_history["task_train_loss"].append(avg_classifier_loss)
        training_history["train_accuracy"].append(classification_accuracy)
        
        
    elapsed = time.perf_counter() - start

    return model, training_history, feature_history, elapsed




def train_nan_cnn_diverse_activations_show_features_localised(  data, 
                    input_dims,
                    n_epochs=100, 
                    batch_size=64,
                    dual_lr = False,
                    learning_rate=0.001, 
                    classifier_lr=0.0001,
                    ae_lr=0.001,
                    n_filters=16,
                    stride=1,
                    padding=1,
                    kernel_size=3,
                    pool_kernel_size=2,
                    pool_stride=2,
                    n_classes=10,
                    epochs_to_show=[1],
                    lambda_cosine=0.2,
                    seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "recon_train_loss": [],
    "sim_train_loss": [],
    "task_train_loss": [],
    "train_accuracy": []
    }

    feature_history = {}
        

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

    #image to track across the epochs
    probe_image, probe_label = data[0]
    probe_image = probe_image.unsqueeze(0).to(device)
        

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)

    #when using one singular learning rate for the optimisers
    if dual_lr == False:
         ae_lr = learning_rate
         classifier_lr = learning_rate
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=ae_lr
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=classifier_lr)


    for epoch in range(n_epochs):

        if (epoch) in epochs_to_show:

            with torch.no_grad():

                #get the unpooled feature maps
                maps = model.feature_maps(probe_image)

                #get the pooled feature maps before flattening
                pooled_maps = model.pool(maps)

                #get the reconstruction for every filter
                recons = torch.cat(
                    [f(probe_image) for f in model.filters],
                    dim=1
                )

                #encoder weights
                weights = torch.stack([
                    f.encoder.weight.squeeze().cpu().clone()
                    for f in model.filters
                ])

                #decoder weights 
                decoder_weights = torch.stack([
                f.decoder.weight.squeeze().cpu().clone()
                for f in model.filters
            ])
                #make the prediction
                logits = model(probe_image)
                prediction = logits.argmax(1).item()

                feature_history[epoch] = {
                    "label": probe_label,
                    "prediction":prediction,
                    "logits": logits.cpu().clone(),
                    "original": probe_image.cpu().clone(),
                    "maps": maps.cpu().clone(),
                    "pooled_maps": pooled_maps.cpu().clone(),
                    "reconstructions": recons.cpu().clone(),
                    "encoder_weights": weights.cpu(),
                    "decoder_weights": decoder_weights.cpu()

                }

        encoder_epoch_loss =0.0
        recon_epoch_loss=0.0
        similarity_epoch_loss=0.0
        classifier_epoch_loss =0.0

        correct = 0
        total = 0

        #per batch
        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            pooled_activations = [
            model.pool(f.encode(images)).flatten(1)
            for f in model.filters
        ]


            # each filter encodes and decodes their input (would be performed in parallel on specialised hardware)
            for j in range(n_filters):


                #pool_map_j = model.pool(model.filters[j].encode(images))

                pool_map_j = pooled_activations[j]


                loss_div = 0

                for k in range(n_filters):

                    if k == j:
                        continue

                    pool_maps_k = pooled_activations[k].detach()


                    loss_div += F.cosine_similarity(
                        pool_map_j,
                        pool_maps_k,
                        dim=1,

                    ).pow(2).mean() #need to take the mean since we are over the batch here
                

                #get the optimiser associeted with filter
                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct(images, j)

                recon_loss = encoder_criterion(x_hat, images)

                # need to also  test similarity = / (n_filters - 1) #since every filter compares itself to n_filters-1 others
                
                similarity = lambda_cosine* loss_div

           

                loss = recon_loss + similarity

                #print(f"recon: {recon_loss} similarity (scaled): {similarity}")

                loss.backward()

                optimizer.step()

                encoder_epoch_loss += loss.item()
                recon_epoch_loss += recon_loss.item()
                similarity_epoch_loss += similarity.item()
   

            classifier_optimizer.zero_grad()

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

        #same for the distinct reconstruction error
        avg_recon_loss = recon_epoch_loss / len(train_loader)
        avg_recon_loss /= n_filters

        #same for the similarity between filters
        avg_similarity_loss = similarity_epoch_loss / len(train_loader)
        avg_similarity_loss /= n_filters


        #for average classification loss, divide by the batch size and then form as percentage
        avg_classifier_loss = classifier_epoch_loss / len(train_loader)
        classification_accuracy = 100.0 * correct / total

        #print and append whole loss term used for training the autoencoder
        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {avg_encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(avg_encoder_loss)

        #print and append the average reconstruction loss for the autoencoder
        print(f"Epoch [{epoch + 1}/{n_epochs}], Average reconstruction Loss: {avg_recon_loss:.4f}")
        training_history["recon_train_loss"].append(avg_recon_loss)

        #print and append the average cosine similarity between encoder filters
        print(f"Epoch [{epoch + 1}/{n_epochs}], Average similarity: {avg_similarity_loss:.4f}")
        training_history["sim_train_loss"].append(avg_similarity_loss)

        #print and append classification performance
        print(f"Epoch [{epoch + 1}/{n_epochs}], Task Training Loss: {avg_classifier_loss:.4f}, Accuracy: {classification_accuracy:.2f}%")

        training_history["task_train_loss"].append(avg_classifier_loss)
        training_history["train_accuracy"].append(classification_accuracy)
        
        
    elapsed = time.perf_counter() - start

    return model, training_history, feature_history, elapsed










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


def crelu_train_nan_cnn(  data, 
                    input_dims,
                    n_epochs=100, 
                    batch_size=64,
                    dual_lr = False,
                    learning_rate=0.001, 
                    classifier_lr=0.0001,
                    ae_lr=0.001,
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
    model = crelu_nan_cnn.FilterCNN(
        input_dims=input_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        bias=bias,
        classes=n_classes
    ).to(device)
    

    #autoencoding loss is MSE of reconstruction vs input
    encoder_criterion = torch.nn.MSELoss().to(device)

    #classifier loss is cross entropy
    classifier_criterion = torch.nn.CrossEntropyLoss().to(device)

    #when using one singular learning rate for the optimisers
    if dual_lr == False:
         ae_lr = learning_rate
         classifier_lr = learning_rate
  
    # separate optimisers are stored per filter, where each filter's parameters span the encoding and decoding weights and biases
    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=ae_lr
        )
        for j in range(n_filters)
    ]

    #classifier optimiser only adjusts weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=classifier_lr)


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
