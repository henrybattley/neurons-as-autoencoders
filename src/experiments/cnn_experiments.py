import os
import numpy as np
import torch, torchvision
import random

#standard cnn
from src.models import cnn_model
#standard cnn without pooling operation
from src.models import cnn_model_no_pooling
#autoencoder using convolution
from src.models import conv_autoencoder
#same convolutional autoencoder but with tied weights for the autoencoding
from src.models import weight_tied_conv_autoencoder
#same model (tied weights) without pooling operation
from src.models import no_pool_weight_tied_conv_autoencoder

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
                in_channels=1,
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
        in_channels=in_channels,
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
                in_channels=1,
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
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        pool_padding=0,
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
                in_channels=1,
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                n_filters=16,
                stride=1,
                padding=1,
                kernel_size=3,
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
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
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



def train_annelaing_cnn(  
                train_data, 
                test_data, 
                input_dims,
                seed_worker,
                n_epochs=30, 
                batch_size=64,
                learning_rate=0.001,
                n_filters=16,
                stride=1,
                padding=1,
                kernel_size=3,
                pool_kernel_size=2,
                pool_stride=2,
                pool_padding=0,
                in_channels=1,
                n_classes=10,
                bias=True,
                seed=42):
    

    training_history = {
    "train_loss": [],
    "train_accuracy":[],
    "test_loss": []
    }

    test_history = {
    "task_loss": [],
    "task_accuracy": []
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
    train_data,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    num_workers=2,
    worker_init_fn=seed_worker,  
    pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    generator=g,
    num_workers=2,
    worker_init_fn=seed_worker,  
    pin_memory=True,
    )


    model = cnn_model.CNN(
        input_dims=input_dims,
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        pool_padding = pool_padding,
        classes=n_classes,
        bias=bias
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    #optimiser and schedular as defined in experiment_hebbian.py
    sup_optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=learning_rate
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        sup_optimizer,
        lr_lambda=lambda epoch:
            0.5 ** max(0, (epoch - 10) // 2)
        )

    
    for epoch in range(n_epochs):

        #train
        train_loss,accuracy = global_backprop.train(model, train_loader, criterion, sup_optimizer, device)

        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(accuracy)

        #test
        with torch.no_grad():

            test_epoch_loss = 0.0
            test_correct = 0
            test_total_samples = 0

            for inputs, labels in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_epoch_loss += loss.detach() 

                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum()
                test_total_samples += labels.size(0)

        avg_test_loss = test_epoch_loss.item() / len(test_loader)
        test_accuracy = 100.0 * test_correct.item() / test_total_samples

        
        test_history["task_loss"].append(avg_test_loss)
        test_history["task_accuracy"].append(test_accuracy)

        #update the scheduler
        scheduler.step()

        current_lr = sup_optimizer.param_groups[0]["lr"]


        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {accuracy:.2f}, Task Test Loss: {avg_test_loss}, Task Test Accuracy: {test_accuracy}% LR: {current_lr:.6f}")


    elapsed = time.perf_counter() - start

    return model, training_history,test_history,elapsed





def train_ae_cnn(  data, 
                input_dims,
                in_channels=1,
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
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        classes=n_classes,
        bias=bias
    ).to(device)



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

            encoder_epoch_loss+=encoder_loss.item()

            #now train the classifier

            classifier_optimizer.zero_grad()
            
            with torch.no_grad():

                features = model.encode(inputs)

            task_outputs = model.classify(features)

            task_loss = task_criterion(task_outputs, labels)
            task_loss.backward()
            classifier_optimizer.step()

            task_epoch_loss += task_loss.item()

            _, predicted = torch.max(task_outputs, 1)
            correct += (predicted == labels).sum()
            total_samples += labels.size(0)

        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        #scale encoder loss to be the same as our filters as autoencoders method?
        #avg_encoder_loss /= n_filters

        avg_task_loss = task_epoch_loss/ len(train_loader)
        accuracy = 100 * correct.item() / total_samples


        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Loss: {avg_encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(avg_encoder_loss)


        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {avg_task_loss:.4f}, Training Accuracy: {accuracy:.2f}")
        training_history["task_train_loss"].append(avg_task_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history, feature_history, elapsed



def train_ae_cnn_weight_tied(  data, 
                input_dims,
                in_channels=1,
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
    "encoder_train_loss": [],


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


    model = weight_tied_conv_autoencoder.CNN_AE(
        input_dims=input_dims,
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        pool_padding=0,
        classes=n_classes,
        bias=bias
    ).to(device)



    encoder_criterion = torch.nn.MSELoss()
    encoder_criterion.to(device)

    task_criterion = torch.nn.CrossEntropyLoss()
    task_criterion.to(device)


    ae_params = [model.encoder.weight]

    if model.encoder.bias is not None:
        ae_params.append(model.encoder.bias)

    if model.decoder_bias is not None:
        ae_params.append(model.decoder_bias)

    ae_optimizer = torch.optim.Adam(ae_params, lr=learning_rate)


    classifier_optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)


    # Training loop..
    for epoch in range(n_epochs):

            
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

            encoder_epoch_loss+=encoder_loss.item()

            #now train the classifier

            classifier_optimizer.zero_grad()
            
            with torch.no_grad():

                features = model.encode(inputs)

            task_outputs = model.classify(features)

            task_loss = task_criterion(task_outputs, labels)
            task_loss.backward()
            classifier_optimizer.step()

            task_epoch_loss += task_loss.item()

            _, predicted = torch.max(task_outputs, 1)
            correct += (predicted == labels).sum()
            total_samples += labels.size(0)

        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        #scale encoder loss to be the same as our filters as autoencoders method?
        #avg_encoder_loss /= n_filters

        avg_task_loss = task_epoch_loss/ len(train_loader)
        accuracy = 100 * correct.item() / total_samples


        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Loss: {avg_encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(avg_encoder_loss)


        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {avg_task_loss:.4f}, Training Accuracy: {accuracy:.2f}")
        training_history["task_train_loss"].append(avg_task_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history, elapsed




def train_ae_cnn_get_features(  data, 
                input_dims,
                in_channels=1,
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
    "task_train_loss": [],
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
        in_channels=in_channels,
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

            encoder_epoch_loss+=encoder_loss.item()

            #now train the classifier

            classifier_optimizer.zero_grad()
            
            with torch.no_grad():

                features = model.encode(inputs)

            task_outputs = model.classify(features)

            task_loss = task_criterion(task_outputs, labels)
            task_loss.backward()
            classifier_optimizer.step()

            task_epoch_loss += task_loss.item()

            _, predicted = torch.max(task_outputs, 1)
            correct += (predicted == labels).sum()
            total_samples += labels.size(0)

        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        #scale encoder loss to be the same as our filters as autoencoders method?
        #avg_encoder_loss /= n_filters

        avg_task_loss = task_epoch_loss/ len(train_loader)
        accuracy = 100 * correct.item() / total_samples


        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Loss: {avg_encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(avg_encoder_loss)


        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {avg_task_loss:.4f}, Training Accuracy: {accuracy:.2f}")
        training_history["task_train_loss"].append(avg_task_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history, feature_history, elapsed


def train_ae_cnn_weight_tie_get_features(  data, 
                input_dims,
                in_channels=1,
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
    "task_train_loss": [],
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


    model = weight_tied_conv_autoencoder.CNN_AE(
        input_dims=input_dims,
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        pool_padding=0,
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

    ae_params = [model.encoder.weight]

    if model.encoder.bias is not None:
        ae_params.append(model.encoder.bias)

    if model.decoder_bias is not None:
        ae_params.append(model.decoder_bias)

    ae_optimizer = torch.optim.Adam(ae_params, lr=learning_rate)

    classifier_optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)


    # Training loop..
    for epoch in range(n_epochs):


        if (epoch) in epochs_to_show:

            with torch.no_grad():

                #get the unpooled feature maps
                maps = model.encode(probe_image)

                pooled_maps = model.pool(maps)

                encoder_weights = model.encoder.weight.detach().cpu().clone()

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
                }



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

            encoder_epoch_loss+=encoder_loss.item()

            #now train the classifier

            classifier_optimizer.zero_grad()
            
            with torch.no_grad():

                features = model.encode(inputs)

            task_outputs = model.classify(features)

            task_loss = task_criterion(task_outputs, labels)
            task_loss.backward()
            classifier_optimizer.step()

            task_epoch_loss += task_loss.item()

            _, predicted = torch.max(task_outputs, 1)
            correct += (predicted == labels).sum()
            total_samples += labels.size(0)

        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        #scale encoder loss to be the same as our filters as autoencoders method?
        #avg_encoder_loss /= n_filters

        avg_task_loss = task_epoch_loss/ len(train_loader)
        accuracy = 100 * correct.item() / total_samples


        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Loss: {avg_encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(avg_encoder_loss)


        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {avg_task_loss:.4f}, Training Accuracy: {accuracy:.2f}")
        training_history["task_train_loss"].append(avg_task_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history, feature_history, elapsed



def train_no_pool_ae_cnn_weight_tied(  data, 
                input_dims,
                in_channels=1,
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                n_filters=16,
                stride=1,
                padding=1,
                output_padding=0,
                kernel_size=3,
                n_classes=10,
                bias=True,
                seed=42):
    

    training_history = {
    "task_train_loss": [],
    "train_accuracy":[],
    "encoder_train_loss": [],
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


    model = no_pool_weight_tied_conv_autoencoder.CNN_AE(
        input_dims=input_dims,
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        n_filters=n_filters,
        classes=n_classes,
        bias=bias
    ).to(device)


    encoder_criterion = torch.nn.MSELoss()
    encoder_criterion.to(device)

    task_criterion = torch.nn.CrossEntropyLoss()
    task_criterion.to(device)


    ae_params = [model.encoder.weight]

    if model.encoder.bias is not None:
        ae_params.append(model.encoder.bias)

    if model.decoder_bias is not None:
        ae_params.append(model.decoder_bias)

    ae_optimizer = torch.optim.Adam(ae_params, lr=learning_rate)


    classifier_optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)


    # Training loop..
    for epoch in range(n_epochs):

            
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

            encoder_epoch_loss+=encoder_loss.item()

            #now train the classifier

            classifier_optimizer.zero_grad()
            
            with torch.no_grad():

                features = model.encode(inputs)

            task_outputs = model.classify(features)

            task_loss = task_criterion(task_outputs, labels)
            task_loss.backward()
            classifier_optimizer.step()

            task_epoch_loss += task_loss.item()

            _, predicted = torch.max(task_outputs, 1)
            correct += (predicted == labels).sum()
            total_samples += labels.size(0)

        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        #scale encoder loss to be the same as our filters as autoencoders method?
        #avg_encoder_loss /= n_filters

        avg_task_loss = task_epoch_loss/ len(train_loader)
        accuracy = 100 * correct.item() / total_samples


        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Loss: {avg_encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(avg_encoder_loss)


        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {avg_task_loss:.4f}, Training Accuracy: {accuracy:.2f}")
        training_history["task_train_loss"].append(avg_task_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history, elapsed



def train_linear_no_pool_ae_cnn_weight_tied(  data, 
                input_dims,
                in_channels=1,
                n_ae_epochs=100, 
                n_classifier_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                n_filters=16,
                stride=1,
                padding=1,
                output_padding=0,
                kernel_size=3,
                n_classes=10,
                bias=True,
                seed=42):
    

    training_history = {
    "task_train_loss": [],
    "train_accuracy":[],
    "encoder_train_loss": [],
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


    model = no_pool_weight_tied_conv_autoencoder.CNN_AE(
        input_dims=input_dims,
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        n_filters=n_filters,
        classes=n_classes,
        bias=bias
    ).to(device)


    encoder_criterion = torch.nn.MSELoss()
    encoder_criterion.to(device)

    task_criterion = torch.nn.CrossEntropyLoss()
    task_criterion.to(device)


    ae_params = [model.encoder.weight]

    if model.encoder.bias is not None:
        ae_params.append(model.encoder.bias)

    if model.decoder_bias is not None:
        ae_params.append(model.decoder_bias)

    ae_optimizer = torch.optim.Adam(ae_params, lr=learning_rate)


    classifier_optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)


    # Training loop..
    for epoch in range(n_ae_epochs):

        model.train()

        encoder_epoch_loss=0.0

        for inputs, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)

            ae_optimizer.zero_grad()

            x_hat = model.autoencode(inputs)
            encoder_loss = encoder_criterion(x_hat,inputs)

            encoder_loss.backward()
            ae_optimizer.step()

            encoder_epoch_loss+=encoder_loss.item()

            
        avg_encoder_loss = encoder_epoch_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{n_ae_epochs}], Encoder Loss: {avg_encoder_loss:.4f}")
        training_history["encoder_train_loss"].append(avg_encoder_loss)



    for epoch in range(n_classifier_epochs):

        model.train()

        task_epoch_loss = 0.0
        correct = 0
        total_samples = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
            classifier_optimizer.zero_grad()

            with torch.no_grad():
            
                features = model.encode(inputs)
            
            task_outputs = model.classify(features)
            
            task_loss = task_criterion(task_outputs, labels)
            task_loss.backward()
            classifier_optimizer.step()
            
            task_epoch_loss += task_loss.item()
            
            _, predicted = torch.max(task_outputs, 1)
            correct += (predicted == labels).sum()
            total_samples += labels.size(0)

        avg_task_loss = task_epoch_loss/ len(train_loader)
        accuracy = 100 * correct.item() / total_samples


        print(f"Epoch [{epoch + 1}/{n_classifier_epochs}], Training Loss: {avg_task_loss:.4f}, Training Accuracy: {accuracy:.2f}")

        training_history["task_train_loss"].append(avg_task_loss)
        training_history["train_accuracy"].append(accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history, elapsed



def train_linear_annealing_ae_cnn_weight_tied(  
                train_data, 
                test_data,
                input_dims,
                seed_worker,
                in_channels=1,
                n_ae_epochs=1, 
                n_classifier_epochs=30, 
                batch_size=64,
                learning_rate=0.001,
                n_filters=16,
                stride=1,
                padding=1,
                kernel_size=3,
                pool_kernel_size=2,
                pool_stride=2,
                pool_padding =0,
                n_classes=10,
                bias=True,
                seed=42):
    

    training_history = {
    "task_train_loss": [],
    "train_accuracy":[],
    "encoder_train_loss": [],


    }
    test_history = {
    "task_loss": [],
    "task_accuracy": []
    }
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is: {device}")

    #seed randomness (akin to the set_seed() function within the hebbian experiment)
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
    train_data,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    num_workers=2,
    worker_init_fn=seed_worker,  
    pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    generator=g,
    num_workers=2,
    worker_init_fn=seed_worker,  
    pin_memory=True,
    )


    model = weight_tied_conv_autoencoder.CNN_AE(
        input_dims=input_dims,
        in_channels=in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_filters=n_filters,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        pool_padding = pool_padding,
        classes=n_classes,
        bias=bias
    ).to(device)


    encoder_criterion = torch.nn.MSELoss()
    encoder_criterion.to(device)

    task_criterion = torch.nn.CrossEntropyLoss()
    task_criterion.to(device)


    ae_params = [model.encoder.weight]

    if model.encoder.bias is not None:
        ae_params.append(model.encoder.bias)

    if model.decoder_bias is not None:
        ae_params.append(model.decoder_bias)

    ae_optimizer = torch.optim.Adam(ae_params, lr=learning_rate)


    # autoencoder training loop..
    for epoch in range(n_ae_epochs):

        model.train()

        encoder_epoch_loss=0.0

        for inputs, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)

            ae_optimizer.zero_grad()

            x_hat = model.autoencode(inputs)
            encoder_loss = encoder_criterion(x_hat,inputs)

            encoder_loss.backward()
            ae_optimizer.step()

            encoder_epoch_loss+=encoder_loss.item()

            
        avg_encoder_loss = encoder_epoch_loss / len(train_loader)


        print(f"Epoch [{epoch + 1}/{n_ae_epochs}], Encoder Loss: {avg_encoder_loss:.4f}")

        training_history["encoder_train_loss"].append(avg_encoder_loss)


    print("Training classifier...")
    #now train the classifier

    #optimiser and schedular as defined in experiment_hebbian.py
    sup_optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=learning_rate
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        sup_optimizer,
        lr_lambda=lambda epoch:
            0.5 ** max(0, (epoch - 10) // 2)
        )

    for epoch in range(n_classifier_epochs):

        task_epoch_loss = 0.0

        task_test_epoch_loss=0.0

        correct = 0
        total_samples = 0

        test_correct = 0
        test_total = 0
            
    
        for inputs, labels in train_loader:

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            #classifier_optimizer.zero_grad()
            sup_optimizer.zero_grad()
            
            with torch.no_grad():

                features = model.encode(inputs)

            task_outputs = model.classify(features)

            task_loss = task_criterion(task_outputs, labels)
            task_loss.backward()

            sup_optimizer.step()

            task_epoch_loss += task_loss.item()

            _, predicted = torch.max(task_outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)


        avg_task_loss = task_epoch_loss/ len(train_loader)
        accuracy = 100 * correct / total_samples

        #now do the eval on the test set
        with torch.no_grad():
            for images, labels in test_loader:
                
                images = images.to(device)
   
                labels = labels.to(device)

                features = model.encode(images)

                task_outputs = model.classify(features)
                
                task_loss = task_criterion(task_outputs, labels)

                task_test_epoch_loss+= task_loss.item()

                _, predicted = torch.max(task_outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)


        avg_test_loss = task_test_epoch_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total


        #update the scheduler
        scheduler.step()

        current_lr = sup_optimizer.param_groups[0]["lr"]



        print(f"Epoch [{epoch + 1}/{n_classifier_epochs}], Task Training Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%, Task Test Loss: {avg_test_loss}, Task Test Accuracy: {test_accuracy}% LR: {current_lr:.6f}")

        training_history["task_train_loss"].append(avg_task_loss)
        training_history["train_accuracy"].append(accuracy)

        
        test_history["task_loss"].append(avg_test_loss)
        test_history["task_accuracy"].append(test_accuracy)


    elapsed = time.perf_counter() - start
    return model, training_history, test_history, elapsed









