import os
import numpy as np
import torch, torchvision
import random
import time

from src.models import nan_cnn

torch.backends.cudnn.benchmark = True



#from src.optimizers import global_backprop
#from src.optimizers import nan_cnn_local_gd

#Training pipeline, when hill_climb=True training follows that as defined by Bull 
def train_nan_cnn(  data, 
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                n_filters=16,
                seed=42):
    
    training_history = {
    "encoder_train_loss": [],
    "task_train_loss": [],
    "train_accuracy": []
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
    num_workers=0,  
    pin_memory=True,
    persistent_workers=True,
)
    


    
 
    #starting time from model definition
    start = time.perf_counter()

    model = nan_cnn.FilterCNN(
        kernel_size=3,
        stride=1,
        padding=1,
        n_filters=n_filters,
        classes=10
    )


    model.to(device)

    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    

    encoder_criterion = torch.nn.MSELoss()
    encoder_criterion.to(device)

    classifier_criterion = torch.nn.CrossEntropyLoss()
    classifier_criterion.to(device)

    filter_optimizers = [
        torch.optim.Adam(
            model.filters[j].parameters(),
            lr=learning_rate
        )
        for j in range(n_filters)
    ]


    #only touch weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=learning_rate)

    for epoch in range(n_epochs):

        encoder_epoch_loss =0.0
        classifier_epoch_loss =0.0

        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            for j in range(n_filters):

                optimizer = filter_optimizers[j]

                optimizer.zero_grad()

                x_hat = model.reconstruct_filter(images, j)

                loss = encoder_criterion(x_hat, images)

                loss.backward()

                optimizer.step()

                encoder_epoch_loss += loss.item()
            

            classifier_optimizer.zero_grad()

            logits = model(images)

            loss = classifier_criterion(logits, labels)

            loss.backward()

            classifier_optimizer.step()

            classifier_epoch_loss += loss.item()

            # Classification accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        

        #divide by the batch size and then the n filters
        avg_encoder_loss = encoder_epoch_loss / len(train_loader)
        avg_encoder_loss /= n_filters



        avg_classifier_loss = classifier_epoch_loss / len(train_loader)
        classification_accuracy = 100.0 * correct / total


    
        #the average reconstruction loss per filter per minibatch.
        print(f"Epoch [{epoch + 1}/{n_epochs}], Encoder Training Loss: {avg_encoder_loss:.4f}")

        training_history["encoder_train_loss"].append(avg_encoder_loss)

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