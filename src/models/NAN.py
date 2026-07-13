import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class NAN(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim:int):
        super().__init__()
        self.input_dim =input_dim
        self.hidden_dim =hidden_dim

        # hidden weight matrix is of shape (H,N) (N is input dim) so each row corresponds to a neurons encoder weights
        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)

        #H x N weights for the decoder (each row again is a neurons decoder weights )
        self.decoder_weight = nn.Parameter(torch.empty(hidden_dim, input_dim))

        #H x N biases for the decoder (easier to think of as a bias per input dim per hidden dim)
        self.decoder_bias = nn.Parameter(torch.empty(hidden_dim, input_dim))

        #regressor head
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
    

    def encode_single(self, x, neuron):
        """
        Returns the encoding produced by a single hidden neuron.

        Shape: (batch_size, 1)
        """

        w = self.hidden.weight[neuron].unsqueeze(0)   # (1,input_dim)
        b = self.hidden.bias[neuron]                  # scalar

        return self.activation(x @ w.T + b)
    
    
    def reconstruct_single(self, x, neuron):

        #each neuron encodes the input (h is the matrix of encodings of shape (N,H)- a row per sample and col per hidden neuron)
        #h = self.encode(x)

        #returns the 2D vector of all N encodings for a neuron (e. g if we indexed h[:, neuron] this would give 1D but we need 2D for the broadcasting we do)
        #h_j = h[:, neuron:neuron+1] #gives shape (n_samples,1)

        h_j = self.encode_single(x, neuron)


        #since every decoder weight belongs to a neuron we simply index the decoder weights by the neuron selected (the row in the decoder weight matrix corresponds to a neuron)
        w = self.decoder_weight[neuron].unsqueeze(0) #again we get the 2D vector (using unsqueeze) for later broadcasting, shape is (1,N)
        #same for the bias (one bias per neuron)
        b = self.decoder_bias[neuron].unsqueeze(0)


        # now we do the broadcasting- every scalar activation (encoding) gets multiplied by the N decoder weights
        return self.activation(h_j * w + b) # so this is the reconstruction by a single neuron from the latent code back to the input space (h_j * w gives shape (n_samples, N))
        #return h_j * w 


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

""" hidden weight (i, j)belongs to hidden neuron i
Hidden bias i belongs to hidden neuron i
decoder weight (i, j) belongs to hidden neuron i
Decoder bias (i, j) belongs to hidden neuron i """
