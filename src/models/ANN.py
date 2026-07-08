import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class ANN(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim:int):
        super().__init__()
        self.input_dim =input_dim
        self.hidden_dim =hidden_dim

        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)

        self.decoder = nn.Linear(self.hidden_dim,self.input_dim)

        self.output = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        # initialization as per Bull's paper (although 0-ing the bias initially is better practise, I'm unsure if Larry has a random range for the bias or not?)
        nn.init.uniform_(self.hidden.weight, -1.0, 1.0)
        nn.init.uniform_(self.hidden.bias, -1.0, 1.0)

        nn.init.uniform_(self.decoder.weight, -1.0, 1.0)
        nn.init.uniform_(self.decoder.bias, -1.0, 1.0)

        nn.init.uniform_(self.output.weight, -1.0, 1.0)
        nn.init.uniform_(self.output.bias, -1.0, 1.0)

        
    def forward(self, x):

        #latent code
        x = self.activation(self.hidden(x))

        #reconstructed input
        x_hat = self.activation(self.decoder(x))

        #regressor head
        y = self.activation(self.output(x))


        return x_hat,y