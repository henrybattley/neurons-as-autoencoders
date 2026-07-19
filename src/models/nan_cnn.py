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

    #encode input (used by individual filters)
    def encode(self, x):

        h = self.activation(self.encoder(x))

        return h
    
    #decode the laten feature representation (used by individual filters)
    def reconstruct(self, x):

        h = self.encode(x)

        x_hat = torch.sigmoid(self.decoder(h))

        return x_hat
    
    
class FilterCNN(nn.Module):

        
    def __init__(self, kernel_size:int, stride: int, padding:int, n_filters:int,classes:int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding =padding
        self.n_filters = n_filters
        

        self.filters = nn.ModuleList([ConvFilter(kernel_size,stride,padding)for _ in range(n_filters)])

        #self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool2d(2,2)

        #the classifier output how do we parametize these dimensions to change with input params
        self.fc = nn.Linear(n_filters * 14 * 14,classes)

    
    # NEED TO ADD SOME DETACH LOGIC HERE SUCH THAT GRADIENTS FOR THE FILTERS ARE NOT COMPUTED (ALTHOUGH OUR OPTIMISER DOESN'T UPDATE THOSE PARAMS, THE GRADS COULD STILL BE COMPUTED?)?
    def forward(self,x):

        feature_maps = []

        for filt in self.filters:

            h = filt.encode(x)

            feature_maps.append(h)

        features = torch.cat(feature_maps, dim=1)

        features = self.pool(features)

        #flatten
        features = features.view(features.size(0), -1)


        return self.fc(features)
    

    # local reconstruction of ONE filter
    def reconstruct_filter(self, x, filter_idx):

        return self.filters[filter_idx].reconstruct(x)
    

    def extract_features(self, x):

        feature_maps = []

        for filt in self.filters:
            feature_maps.append(filt.encode(x))

        features = torch.cat(feature_maps, dim=1)
        features = self.pool(features)
        features = features.view(features.size(0), -1)

        return features
    

    def classify(self, features):

        return self.fc(features)


    """prior to my bug fixing:
    
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

    #encode input (used by individual filters)
    def encode(self, x):

        h = self.activation(self.encoder(x))

        return h
    
    #decode the laten feature representation (used by individual filters)
    def reconstruct(self, x):

        h = self.encode(x)

        x_hat = torch.sigmoid(self.decoder(h))

        return x_hat
    
    
class FilterCNN(nn.Module):

        
    def __init__(self, kernel_size:int, stride: int, padding:int, n_filters:int,classes:int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding =padding
        self.n_filters = n_filters
        

        self.filters = nn.ModuleList([ConvFilter(kernel_size,stride,padding)for _ in range(n_filters)])

        #self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool2d(2,2)

        #the classifier output how do we parametize these dimensions to change with input params
        self.fc = nn.Linear(n_filters * 14 * 14,classes)

    
    # NEED TO ADD SOME DETACH LOGIC HERE SUCH THAT GRADIENTS FOR THE FILTERS ARE NOT COMPUTED (ALTHOUGH OUR OPTIMISER DOESN'T UPDATE THOSE PARAMS, THE GRADS COULD STILL BE COMPUTED?)?
    def forward(self,x):

        feature_maps = []

        for filt in self.filters:

            h = filt.encode(x)

            feature_maps.append(h)

        features = torch.cat(feature_maps, dim=1)

        features = self.pool(features)

        #flatten
        features = features.view(features.size(0), -1)


        return self.fc(features)
    

    # local reconstruction of ONE filter
    def reconstruct_filter(self, x, filter_idx):

        return self.filters[filter_idx].reconstruct(x)"""

