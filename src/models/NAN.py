import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class NAN(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim:int):
        super().__init__()
        self.input_dim =input_dim
        self.hidden_dim =hidden_dim

        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)

        #self.decoder = nn.Linear(self.hidden_dim,self.input_dim)

        #H x N weights for the decoder
        self.decoder_weight = nn.Parameter(torch.empty(hidden_dim, input_dim))

        #H x N biases for the decoder (a bias per input dim per hidden dim)
        self.decoder_bias = nn.Parameter(torch.empty(hidden_dim, input_dim))

        self.output = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        # initialization as per Bull's paper (although 0-ing the bias initially is better practise, I'm unsure if Larry has a random range for the bias or not?)
        nn.init.uniform_(self.hidden.weight, -1.0, 1.0)
        nn.init.uniform_(self.hidden.bias, -1.0, 1.0)

        nn.init.uniform_(self.decoder_weight, -1.0, 1.0)
        nn.init.uniform_(self.decoder_bias, -1.0, 1.0)

        nn.init.uniform_(self.output.weight, -1.0, 1.0)
        nn.init.uniform_(self.output.bias, -1.0, 1.0)


    def encode(self, x):
        return self.activation(self.hidden(x))
    
    
    def reconstruct_single(self, x, neuron):

        h = self.encode(x)

        h_j = h[:, neuron:neuron+1]

        #w = self.decoder.weight[:, neuron].unsqueeze(0)
        #b = self.decoder.bias.unsqueeze(0)

        w = self.decoder_weight[neuron].unsqueeze(0)
        b = self.decoder_bias[neuron].unsqueeze(0)



        #return self.activation(h_j * w + b)
        return h_j * w 


    def regress(self, x):

        h = self.encode(x)

        return self.activation(self.output(h))
    

    def forward(self, x):

        return self.regress(x)

""" 
        #latent code
        #x = self.activation(self.hidden(x))

        #reconstructed input
        #x_hat = self.activation(self.decoder(x))

        #regressor head
        #y = self.activation(self.output(x))

        x = self.encode(x)

        #x_hat = self.activation(self.decoder(x))

        y = self.activation(self.output(x))


        return y
    """
