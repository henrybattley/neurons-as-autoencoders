import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

"""
--Neuron-based autoencoder as described by Bull--
-Sigmoid activations for every node
-Biases for every node
"""

""" this looks conceptually cleaner, however it isn't that vectorized... we have to loop over the hidden nodes to perform the encoding for the regressor (in our other version this is a vector operation)"""

class NANNeuron(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.encoder_weight = nn.Parameter(torch.empty(input_dim))
        self.encoder_bias   = nn.Parameter(torch.empty(1))

        self.decoder_weight = nn.Parameter(torch.empty(input_dim))
        self.decoder_bias   = nn.Parameter(torch.empty(input_dim))

        nn.init.uniform_(self.encoder_weight, -1, 1)
        nn.init.uniform_(self.encoder_bias, -1, 1)

        nn.init.uniform_(self.decoder_weight, -1, 1)
        nn.init.uniform_(self.decoder_bias, -1, 1)

        self.activation = nn.Sigmoid()
        
    

class refactor_NAN_GD(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim:int):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.neurons = nn.ModuleList(
            [
                NANNeuron(input_dim)
                for _ in range(hidden_dim)
            ]
        )

        self.output = nn.Linear(hidden_dim,1)

        self.activation = nn.Sigmoid()


        

    def encode_single(self, x):

        h = self.activation(
            x @ self.encoder_weight.unsqueeze(1)
            + self.encoder_bias
        )

        return h
    
    def reconstruct_single(self, x):

        h = self.encode(x)

        x_hat = self.activation(
            h * self.decoder_weight.unsqueeze(0)
            + self.decoder_bias.unsqueeze(0)
        )

        return x_hat
    
    """ then we'd call model.neurons[5].reconstruct_single(x)"""

    """and opt becomes optimizer = Adam(
    model.neurons[j].parameters(),
    lr=learning_rate
)"""
    

    def encode(self,x):

        hidden = []

        for neuron in self.neurons:

            hidden.append(
                neuron.encode(x)
            )

        return torch.cat(hidden,dim=1)
    

    #output layer task is regression (fitness values between [0,1]), thus return is sigmoid of affine of latent code
    def regress(self, x):

        h = self.encode(x)

        return self.activation(self.output(h))



        #self.hidden = nn.Linear(self.input_dim, self.hidden_dim) # hidden weight matrix is of shape (H,N) (N is input dim) so each row corresponds to a neuron's encoder weights
       
        #self.decoder_weight = nn.Parameter(torch.empty(hidden_dim, input_dim)) #H x N weights for the decoder (each row again is a neuron's decoder weights )

        #self.decoder_bias = nn.Parameter(torch.empty(hidden_dim, input_dim)) #H x N biases for the decoder (easier to think of as a bias per input dim per hidden dim)

    """ 
    def encode(self, x):

        activations = []

        for j in range(self.hidden_dim):

            h_j = self.encode_single(x, j)

            activations.append(h_j)

        return torch.cat(activations, dim=1)"""
    

    
    


""" hidden weight (i, j)belongs to hidden neuron i
Hidden bias i belongs to hidden neuron i
decoder weight (i, j) belongs to hidden neuron i
Decoder bias (i, j) belongs to hidden neuron i """
