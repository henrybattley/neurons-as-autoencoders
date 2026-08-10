import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CNN_AE(nn.Module):
    def __init__(self,
                 input_dims,
                 in_channels,
                 kernel_size,
                 stride,
                 padding, 
                 n_filters, 
                 classes,
                 output_padding,
                 bias=True 
    ):
        super(CNN_AE, self).__init__()

        self.input_dims = input_dims
        self.in_channels=in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding =padding
        self.output_padding=output_padding
        self.n_filters = n_filters
        self.classes = classes
        self.bias =bias


        
        # 1st conv block (creates n_filters feature mappings)
        self.encoder = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding,
            bias=bias)
        
        #He initialisation pre relu activation
        nn.init.kaiming_normal_(self.encoder.weight)


        if bias:

            nn.init.zeros_(self.encoder.bias)
            self.decoder_bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("decoder_bias", None)


        #modern standard activation within convolutional networks is relu
        self.activation = nn.ReLU()

        #self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride) 

        #this calculation used for the flattened dims only works with square input..
        conv_dim = ((input_dims + 2*padding - kernel_size) // stride) + 1             
        #pool_dim = ((conv_dim - pool_kernel_size) // pool_stride) + 1        

        self.fc = nn.Linear(n_filters * conv_dim * conv_dim, classes)

        #xavier init for fc
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    #can be called to visualise activations
    def encode(self,x):

        h = self.activation(self.encoder(x))

        return h

        #calls encode and decode the latent feature representation (used by individual filters)
    def autoencode(self, x):

        h = self.encode(x)

        # do the transpose convolution but using the encoder weights
        x_hat = F.conv_transpose2d(
        h,
        weight=self.encoder.weight,
        bias=self.decoder_bias,
        stride=self.stride,
        padding=self.padding,
        output_padding=self.output_padding
    )

        return torch.sigmoid(x_hat)


    def classify(self, h):

        #h_pool = self.pool(h)

        #flatten here
        h_flat = torch.flatten(h, 1)

        pred = self.fc(h_flat)
        return pred




    
