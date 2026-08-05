import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

"""defines each filter (kernel) with the function of encoding and decoding its input"""
class ConvFilter(nn.Module):
    
    def __init__(self, kernel_size=3,stride=1,padding=1, bias=True):

        super().__init__()

        # encoder 
        self.encoder = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias = bias
        )

        #encoder He initialiasion for pre relu gates 
        nn.init.kaiming_normal_(
                                self.encoder.weight,
                                mode="fan_out",
                                nonlinearity="relu"
        )

        # decoder, uses transpose convolution to restore input dimensions
        self.decoder = nn.ConvTranspose2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )       

        #xavier is useful for symmetric activations (like sigmoid)
        nn.init.xavier_normal_(self.decoder.weight)

        if bias == True:
            nn.init.zeros_(self.encoder.bias)
            nn.init.zeros_(self.decoder.bias)


        #modern standard activation within convolutional networks is relu
        self.activation = nn.ReLU()




    #encode input (used by individual filters)
    def encode(self, x):

        z = self.encoder(x)

        h = torch.cat(
            [self.activation(z), self.activation(-z)],
            dim=1
        )

        return h
    
    #calls encode and decode the latent feature representation (used by individual filters)
    def forward(self, x):

        h = self.encode(x)

        #experiment with different activation here-- perhaps no sigmoid
        x_hat = torch.sigmoid(self.decoder(h))

        return x_hat
    
    
"""defines the network of ConvFilters"""
class FilterCNN(nn.Module):

    def __init__(
            self, 
            input_dims:int, 
            kernel_size:int, 
            stride: int, 
            padding:int, 
            n_filters:int,
            classes:int, 
            pool_kernel_size:int, 
            pool_stride:int,
            bias: bool
        ):
        
        super().__init__()

        self.input_dims = input_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding =padding
        self.n_filters = n_filters
        self.classes = classes
        self.pool_kernel_size=pool_kernel_size
        self.pool_stride = pool_stride
        self.bias= bias
        
        #define the list of autoencoder filter submodules 
        self.filters = nn.ModuleList([ConvFilter(kernel_size,stride,padding,bias)for _ in range(n_filters)])

        self.pool = nn.MaxPool2d(pool_kernel_size,pool_stride)

        #only works with square input..
        conv_dim = ((input_dims + 2*padding - kernel_size) // stride) + 1             

        pool_dim = ((conv_dim - pool_kernel_size) // pool_stride) + 1                

        self.fc = nn.Linear(2 * n_filters * pool_dim * pool_dim, classes)

        #xavier init for linear fully connected
        nn.init.xavier_normal_(self.fc.weight)


        nn.init.zeros_(self.fc.bias)


    # local reconstruction of one filter

    def reconstruct(self, x, filter_idx):

        #refers to the individual model forward function which does the encoding and reconstruction
        return self.filters[filter_idx](x)
    
    
    
    def extract_features(self, x):

        feature_maps = [f.encode(x) for f in self.filters]
        features = torch.cat(feature_maps, dim=1)
        features = self.pool(features)
        features = features.flatten(1)
        return features
    

    def classify(self, features):
        return self.fc(features)


    def forward(self, x):
        return self.classify(self.extract_features(x))

    # get complete feature maps for visualisation (before pooling)
    def feature_maps(self, x):
        maps = [f.encode(x) for f in self.filters]
        return torch.cat(maps, dim=1)

