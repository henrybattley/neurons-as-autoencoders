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

        # initialization as per Bull's paper 
        nn.init.uniform_(self.hidden.weight, -1.0, 1.0)
        nn.init.uniform_(self.hidden.bias, -1.0, 1.0)

        nn.init.uniform_(self.output.weight, -1.0, 1.0)
        nn.init.uniform_(self.output.bias, -1.0, 1.0)

        
    def forward(self, x):

        x = self.activation(self.hidden(x))
      
        x = self.activation(self.output(x))


        return x
    
