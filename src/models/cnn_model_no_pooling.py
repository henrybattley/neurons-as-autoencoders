import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CNN(nn.Module):
    def __init__(self,
                 input_dims,
                 kernel_size,
                 stride,
                 padding, 
                 n_filters, 
                 classes,
                 pool_kernel_size,
                 pool_stride,
                 bias=True 
    ):
        super(CNN, self).__init__()

        self.input_dims = input_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding =padding
        self.n_filters = n_filters
        self.classes = classes
        self.pool_kernel_size=pool_kernel_size
        self.pool_stride = pool_stride


        
        # 1st conv block (creates n_filters feature mappings)
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=n_filters,
                               kernel_size=kernel_size,
                               stride=stride, 
                               padding=padding,
                               bias=bias,)
        
        #He initialisation pre relu activation
        nn.init.kaiming_normal_(self.conv1.weight)
        
        if bias == True:
            nn.init.zeros_(self.conv1.bias)


        #this calculation used for the flattened dims only works with square input..
        conv_dim = ((input_dims + 2*padding - kernel_size) // stride) + 1             
      

        self.fc = nn.Linear(n_filters * conv_dim * conv_dim, classes)

        #xavier init for fc
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


    def forward(self, x):
        #x is (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))

        #flatten here
        x = x.view(x.size(0), -1)  

        x = self.fc(x)
        return x
    
