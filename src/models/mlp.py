import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class MLP(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim:int):
        super().__init__()
        self.input_dim =input_dim
        self.hidden_dim =hidden_dim

        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        # initialization as per Bull's paper (although 0-ing the bias initially is better practise, I'm unsure if Larry has a random range for the bias or not?)
        nn.init.uniform_(self.hidden.weight, -1.0, 1.0)
        nn.init.uniform_(self.hidden.bias, -1.0, 1.0)

        nn.init.uniform_(self.output.weight, -1.0, 1.0)
        nn.init.uniform_(self.output.bias, -1.0, 1.0)

        
    def forward(self, x):

        x = self.activation(self.hidden(x))
      
        x = self.activation(self.output(x))


        return x
    
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
        print(f"[OK] Model exported to {filename}") """