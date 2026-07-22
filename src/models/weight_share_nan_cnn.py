import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

"""defines each filter (kernel) with the function of encoding and decoding its input"""
class WeightShareConvFilter(nn.Module):
    
    def __init__(self, kernel_size=3,stride=1,padding=1):

        super().__init__()

        # encoder 
        self.encoder = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True #this is default anyway
        )

        """ 
        # decoder, uses transpose convolution to restore input dimensions
        self.decoder = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )"""

        #self.decoder_bias = nn.Parameter(torch.zeros(1))

        self.stride = stride
        self.padding = padding

        #standard activation within convolutional networks is relu
        self.activation = nn.ReLU()

    #encode input (used by individual filters)
    def encode(self, x):

        h = self.activation(self.encoder(x))

        return h
    
    #calls encode and decode the latent feature representation (used by individual filters)
    
    def forward(self, x):

        h = self.encode(x)

        # do the transpose convolution but using the encoder weights
        x_hat = F.conv_transpose2d(
        h,
        weight=self.encoder.weight,
        bias=None,
        stride=self.stride,
        padding=self.padding,
    )

        return torch.sigmoid(x_hat)

        """ 
        #experiment with different activation here
        x_hat = torch.sigmoid(self.decoder(h))

        return x_hat"""
    
    
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
            pool_stride:int
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
        
        #define the list of autoencoder filter submodules 
        self.filters = nn.ModuleList([WeightShareConvFilter(kernel_size,stride,padding)for _ in range(n_filters)])


        self.pool = nn.MaxPool2d(pool_kernel_size,pool_stride)

        #only works with square input..
        conv_dim = ((input_dims + 2*padding - kernel_size) // stride) + 1             

        pool_dim = ((conv_dim - pool_kernel_size) // pool_stride) + 1                


        #the classifier output how do we parametize these dimensions to change with input params
        #self.fc = nn.Linear(n_filters * 14 * 14, classes)
        self.fc = nn.Linear(n_filters * pool_dim * pool_dim, classes)

    

    # local reconstruction of one filter

    def reconstruct(self, x, filter_idx):

        #we are now referring to the individual model forward which does the reconstruction
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
