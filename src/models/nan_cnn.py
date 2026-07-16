import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

class ConvFilter(nn.Module):

    
    def __init__(self, kernel_size=3,stride=1,padding=1):

        super().__init__()

        # encoder
        self.encoder = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # decoder
        self.decoder = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.activation = nn.ReLU()

    #encoder_single
    def encode_single(self, x):

        h = self.activation(self.encoder(x))

        return h
    
    def reconstruct_single(self, x):

        h = self.encode_single(x)

        x_hat = torch.sigmoid(self.decoder(h))

        return x_hat
    
    
class FilterCNN(nn.Module):

        
    def __init__(self, kernel_size:int, stride: int, padding:int, n_filters:int,classes:int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding =padding
        self.n_filters = n_filters
        
        #self.filters = nn.ModuleList([ConvFilter() for _ in range(n_filters)])

        self.filters = nn.ModuleList([ConvFilter(kernel_size,stride,padding)for _ in range(n_filters)])

        #self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool2d(2,2)

        #the classifier output how do we parametize these dimensions to change with input params
        self.fc = nn.Linear(n_filters * 14 * 14,classes)

    
    def forward(self,x):

        feature_maps = []

        for filt in self.filters:

            h = filt.encode_single(x)

            feature_maps.append(h)

        features = torch.cat(feature_maps, dim=1)

        features = self.pool(features)

        features = features.view(features.size(0), -1)


        return self.fc(features)
    

    # local reconstruction of ONE filter
    def reconstruct_filter(self, x, filter_idx):

        return self.filters[filter_idx].reconstruct_single(x)

