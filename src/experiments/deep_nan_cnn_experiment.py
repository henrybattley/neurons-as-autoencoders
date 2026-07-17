import os
import numpy as np
import torch, torchvision
import random
import time

from src.models import deep_nan_cnn




#from src.optimizers import global_backprop
#from src.optimizers import nan_cnn_local_gd

#Training pipeline, when hill_climb=True training follows that as defined by Bull 
def train_deep_nan_cnn(  data, 
                n_epochs=100, 
                batch_size=64,
                learning_rate=0.001,
                n_filters=16,
                depth=10,
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
    num_workers=2,  
    pin_memory=True,
    persistent_workers=True,
)
 
    #starting time from model definition
    start = time.perf_counter()

    model = deep_nan_cnn.FilterCNN(
        depth=depth,
        kernel_size=3,
        stride=1,
        padding=1,
        n_filters=n_filters,
        classes=10
    )


    model.to(device)
    

    encoder_criterion = torch.nn.MSELoss()
    encoder_criterion.to(device)

    classifier_criterion = torch.nn.CrossEntropyLoss()
    classifier_criterion.to(device)

    filter_optimizers = [
    [
        torch.optim.Adam(
            model.blocks[b].filters[j].parameters(),
            lr=learning_rate
        )
        for j in range(n_filters)
    ]
    for b in range(depth)
]


    #only touch weights of the fully connected layer
    classifier_optimizer = torch.optim.Adam(model.fc.parameters(),lr=learning_rate)

    for epoch in range(n_epochs):

        encoder_epoch_loss =0.0
        classifier_epoch_loss =0.0

        correct = 0
        total = 0

        for images, labels in train_loader:

            x = images.to(device)
            labels = labels.to(device)

            for b in range(depth):

                block_input = x.detach().clone()

                for j in range(n_filters):

                    optimizer = filter_optimizers[b][j]

                    optimizer.zero_grad()

                    x_hat = model.reconstruct_filter(
                        block_input,
                        block_idx=b,
                        filter_idx=j
                    )

                    #x_hat = model.reconstruct_filter(x, block_idx=b,filter_idx=j)


                    loss = encoder_criterion(x_hat, block_input)

                    loss.backward()

                    optimizer.step()

                    encoder_epoch_loss += loss.item()

                x = model.blocks[b](x)

            for block in model.blocks:
                for p in block.parameters():
                    p.requires_grad_(False)
            

            classifier_optimizer.zero_grad()

            logits = model(images.to(device))

            loss = classifier_criterion(logits, labels)

            loss.backward()

            classifier_optimizer.step()

            classifier_epoch_loss += loss.item()

            # Classification accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for block in model.blocks:
                for p in block.parameters():
                    p.requires_grad_(True)

        

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


"""test dims:

if epoch == 0:

    print(images.shape)

    x = images

    for i, block in enumerate(model.blocks):

        x = block(x)

        print(f"Block {i}: {x.shape}")

    pooled = model.pool(x)

    print(pooled.shape)"""