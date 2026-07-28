import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class GroupedLocalAutoencoders(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, n_filters=16, classes=10):

        super().__init__()
        self.n_filters = n_filters

        # Grouped Encoder: In=N, Out=N, Groups=N 
        # Means: Channel 0 ONLY goes to Filter 0, Channel 1 ONLY to Filter 1, etc.
        self.encoder = nn.Conv2d(
            in_channels=n_filters, 
            out_channels=n_filters, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            groups=n_filters  # Fully isolates each encoder filter!
        )

        """ 
        #encoder He initialiasion for pre relu gates 
        nn.init.kaiming_normal_(
                                self.encoder.weight,
                                mode="fan_out",
                                nonlinearity="relu"
        )
        nn.init.zeros_(self.encoder.bias)"""
        with torch.no_grad():
            # Calculate std for a single filter (out_channels=1 per group)
            fan_out = 1 * kernel_size * kernel_size  # 1 * 3 * 3 = 9
            std = (2.0 / fan_out) ** 0.5
            self.encoder.weight.normal_(0, std)
            nn.init.zeros_(self.encoder.bias)

        # Grouped Decoder: In=N, Out=N, Groups=N
        # Means: Feature Map 0 ONLY goes to Decoder 0, etc.
        self.decoder = nn.ConvTranspose2d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=n_filters  # Fully isolates each decoder filter!
        )

        
        with torch.no_grad():
            # Single filter fan_in and fan_out are both 1 * K * K = 9
            fan_in = 1 * kernel_size * kernel_size
            fan_out = 1 * kernel_size * kernel_size
            
            # Xavier normal std = gain * sqrt(2 / (fan_in + fan_out))
            # gain for Sigmoid / default is 1.0
            std_decoder = (2.0 / (fan_in + fan_out)) ** 0.5
            
            self.decoder.weight.normal_(0, std_decoder)
            nn.init.zeros_(self.decoder.bias)

        """ 
        #xavier is useful for symmetric activations (like sigmoid)
        nn.init.xavier_normal_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)"""

        self.activation = nn.ReLU()

        #change to parameterise
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(n_filters * 14 * 14, classes)

 
        #xavier init for linear fully connected
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)




    def reconstruct_all(self, x):
        # x shape: (Batch, 1, H, W) -> expand to (Batch, N, H, W)
        #duplicates the single image channel n_filter times (so each grouped grouped encoder works with its own copy of every image)
        x_expanded = x.repeat(1, self.n_filters, 1, 1)
        h = self.activation(self.encoder(x_expanded))
        x_hat = torch.sigmoid(self.decoder(h))
        return x_hat, x_expanded, h

    def extract_features(self, x):
        x_expanded = x.repeat(1, self.n_filters, 1, 1)
        h = self.activation(self.encoder(x_expanded))
        features = self.pool(h)
        return features.view(features.size(0), -1)

    def classify(self, features):
        return self.fc(features)

    def forward(self, x):
        features = self.extract_features(x)
        return self.classify(features)





 
"""defines the network of ConvFilters"""
""" 
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
        self.filters = nn.ModuleList([ConvFilter(kernel_size,stride,padding)for _ in range(n_filters)])


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
""" 