import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions



class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # 1st conv block (16 feature mappings but same spatial size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,kernel_size=3,stride=1, padding=1)
        #He initialiasion for all blocks
        nn.init.kaiming_normal_(self.conv1.weight)

        #2x2 pooling used at first two layers (reduces spacial size but keeps depth)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #pooling with strid 2 halfs the number of dimensions


        #fully connected layer has flattened input
        self.fc = nn.Linear(16 * 14 * 14, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)

        
    def forward(self, x):
        #x is (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool(x)


        #flatten here
        x = x.view(x.size(0), -1)  

        x = self.fc(x)
        return x
    
    
    
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return F.relu(self.conv(x))


class DeeperCNN(nn.Module):

    def __init__(
        self,
        depth=10,
        n_filters=16,
        num_classes=10
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        # First convolution
        self.blocks.append(
            ConvBlock(
                in_channels=1,
                out_channels=n_filters
            )
        )

        # Remaining convolutions
        for _ in range(depth - 1):

            self.blocks.append(
                ConvBlock(
                    in_channels=n_filters,
                    out_channels=n_filters
                )
            )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(
            n_filters * 14 * 14,
            num_classes
        )

        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        return self.fc(x)






    """ 
    Final model: Simple 3-layer CNN for 28x28 grayscale images.
    
    Architecture overview:
    - Conv1: 1->16 channels, kernel 3x3, padding 1 -> ReLU -> MaxPool2d(2x2)
      (spatial size: 28x28 -> 14x14)
    - Conv2: 16->32 channels, kernel 3x3, padding 1 -> ReLU -> MaxPool2d(2x2)
      (spatial size: 14x14 -> 7x7)
    - Conv3: 32->64 channels, kernel 3x3, padding 1 -> ReLU
      (spatial size remains 7x7 as no pooling)
    - flatten -> Fully connected layer -> 10 outputs for classes
    
    - He initialization applied to all conv and linear layers
    - pooling reduces only after first two conv layers
    """
    """ 
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # 1st conv block (16 feature mappings but same spatial size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,kernel_size=3, padding=1)
        #He initialiasion for all blocks
        nn.init.kaiming_normal_(self.conv1.weight)

        #2x2 pooling used at first two layers (reduces spacial size but keeps depth)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #pooling with strid 2 halfs the number of dimensions

        # 2nd conv block (increase feature mappings to 32 for richer feature capture, keep spatial size)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)


        #fully connected layer has flattened input
        self.fc = nn.Linear(32 * 7 * 7, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)

        
    def forward(self, x):
        #x is (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        #flatten here
        x = x.view(x.size(0), -1)  

        x = self.fc(x)
        return x
"""


""" 
    #function to get the model out to onnx format for visualisation
    def export_onnx(self, filename="cnn.onnx"):
        dummy = torch.randn(1, 1, 28, 28)
        torch.onnx.export(
            self,
            dummy,
            filename,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=11
        )
        print(f"[OK] Model exported to {filename}")
"""