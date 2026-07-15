import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

"""
--Neuron-based autoencoder as described by Bull--
-Sigmoid activations for every node
-Biases for every node
"""

class NAN_GD(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim:int):
        super().__init__()

        self.input_dim =input_dim
        self.hidden_dim =hidden_dim

        #self.hidden = nn.Linear(self.input_dim, self.hidden_dim) # hidden weight matrix is of shape (H,N) (N is input dim) so each row corresponds to a neuron's encoder weights
       
        #self.decoder_weight = nn.Parameter(torch.empty(hidden_dim, input_dim)) #H x N weights for the decoder (each row again is a neuron's decoder weights )

        #self.decoder_bias = nn.Parameter(torch.empty(hidden_dim, input_dim)) #H x N biases for the decoder (easier to think of as a bias per input dim per hidden dim)


        self.encoder_weights = nn.ParameterList([
        nn.Parameter(torch.empty(input_dim))
        for _ in range(hidden_dim)
        ])

        self.encoder_biases = nn.ParameterList([
            nn.Parameter(torch.empty(1))
            for _ in range(hidden_dim)])
        
        self.decoder_weights = nn.ParameterList([
        nn.Parameter(torch.empty(input_dim))
        for _ in range(hidden_dim)
        ])

        self.decoder_biases = nn.ParameterList([
        nn.Parameter(torch.empty(input_dim))
        for _ in range(hidden_dim)
        ])


        self.output = nn.Linear(self.hidden_dim, 1) #regressor head

        self.activation = nn.Sigmoid()

        # initialization as per Bull's paper 
        for j in range(hidden_dim):
            nn.init.uniform_(self.encoder_weights[j], -1.0, 1.0)
            nn.init.uniform_(self.encoder_biases[j], -1.0, 1.0)

            nn.init.uniform_(self.decoder_weights[j], -1.0, 1.0)
            nn.init.uniform_(self.decoder_biases[j], -1.0, 1.0)
    
    """ 
    #fully connected input connections' affine transform and activation (the feature encoding for regression head)
    def encode(self, x):
        # returns hidden activations (of shape (batch_size, H)) (each neuron encodes the input- return is the matrix of encodings of shape (N,H)- a row per sample and col per hidden neuron)
        
        #return self.activation(self.encoder_weights*x.T+self.encoder_biases)
        
        return self.activation(self.hidden(x))"""

    def encode(self, x):

        activations = []

        for j in range(self.hidden_dim):

            h_j = self.encode_single(x, j)

            activations.append(h_j)

        return torch.cat(activations, dim=1)

    
    #singular selected neuron affine transform and activation (neuron level encoding)
    def encode_single(self, x, neuron):

        #w = self.hidden.weight[neuron].unsqueeze(0)  
        #b = self.hidden.bias[neuron] 

        w = self.encoder_weights[neuron].unsqueeze(0)
        b = self.encoder_biases[neuron]
                       
        return self.activation(x @ w.T + b)
    
    
    def reconstruct_single(self, x, neuron):

        #the latent representation for the neuron selected for perturbation (of size (batch_size,1)
        h_j = self.encode_single(x, neuron)

        #since every decoder weight belongs to a neuron we simply index the decoder weights by the neuron selected (the row in the decoder weight matrix corresponds to a neuron)
        #w = self.decoder_weight[neuron].unsqueeze(0) 
        #same for the bias 
        #b = self.decoder_bias[neuron].unsqueeze(0)

        w = self.decoder_weights[neuron].unsqueeze(0)
        b = self.decoder_biases[neuron].unsqueeze(0)


        #every scalar activation (encoding) gets multiplied by the N decoder weights and the N biases are added on
        return self.activation(h_j * w + b) # so this is the reconstruction by a single neuron from the latent code back to the input space (h_j * w gives shape (n_samples, N))

    #output layer task is regression (fitness values between [0,1]), thus return is sigmoid of affine of latent code
    def regress(self, x):

        h = self.encode(x)

        return self.activation(self.output(h))
    


""" hidden weight (i, j)belongs to hidden neuron i
Hidden bias i belongs to hidden neuron i
decoder weight (i, j) belongs to hidden neuron i
Decoder bias (i, j) belongs to hidden neuron i """
