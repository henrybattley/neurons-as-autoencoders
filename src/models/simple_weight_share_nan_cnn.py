import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

"""defines each filter (kernel) with the function of encoding and decoding its input"""
class SimpleWeightShareConvFilter(nn.Module):
    
    def __init__(self, kernel_size=3,stride=1,padding=1,output_padding=0):

        super().__init__()

        # encoder 
        self.encoder = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False #no encoder bias
        )

        #encoder He initialiasion for pre relu gates 
        nn.init.kaiming_normal_(
                                self.encoder.weight,
                                mode="fan_out",
                                nonlinearity="relu"
        )

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding


        #modern standard activation within convolutional networks is relu
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
        output_padding=self.output_padding
    )

        #maybe we don't even sigmoid here? to simplify
        return torch.sigmoid(x_hat)


    
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
            output_padding:int
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
        self.output_padding =output_padding
        
        #define the list of autoencoder filter submodules 
        self.filters = nn.ModuleList([SimpleWeightShareConvFilter(kernel_size,stride,padding)for _ in range(n_filters)])

        self.pool = nn.MaxPool2d(pool_kernel_size,pool_stride)

        #only works with square input..
        conv_dim = ((input_dims + 2*padding - kernel_size) // stride) + 1             

        pool_dim = ((conv_dim - pool_kernel_size) // pool_stride) + 1                

        self.fc = nn.Linear(n_filters * pool_dim * pool_dim, classes)

        #xavier init for linear fully connected
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


    # local reconstruction of one filter
    def reconstruct(self, x, filter_idx):

        #refers to the individual model forward function which does the reconstruction
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
