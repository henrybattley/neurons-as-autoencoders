import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

"""
--Layer-based autoencoder as described by Bull--
-Initialisation consistent with the Neuron-based autoencoder
-Sigmoid activations for every node
-Biases for every node
"""
class ANN(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim:int):
        super().__init__()
        self.input_dim =input_dim
        self.hidden_dim =hidden_dim

        self.hidden = nn.Linear(self.input_dim, self.hidden_dim) #weights are hidden_dim x input_dim (so a neuron corresponds to a row)
                                                                 #biases are of vector length hidden_dim (so every neuron has one bias from the input connections)

        self.decoder = nn.Linear(self.hidden_dim,self.input_dim) #weights are input_dim x hidden_dim (so a neuron corresponds to a column)
                                                                 #biases are of vector length input_dim (so biases are shared across the hidden dims)
                                                                 #shared biases means when a hidden node is selected for pertubation, any one of the biases may be perturbed (as are considered as belonging to every neuron)

        self.output = nn.Linear(self.hidden_dim, 1) #regressor output (weights correspond to hidden_dim + a bias)

        self.activation = nn.Sigmoid()

        # initialization as per Bull's paper 
        nn.init.uniform_(self.hidden.weight, -1.0, 1.0)
        nn.init.uniform_(self.hidden.bias, -1.0, 1.0)

        nn.init.uniform_(self.decoder.weight, -1.0, 1.0)
        nn.init.uniform_(self.decoder.bias, -1.0, 1.0)

        nn.init.uniform_(self.output.weight, -1.0, 1.0)
        nn.init.uniform_(self.output.bias, -1.0, 1.0)

    #fully connected input connections' affine transform and activation (the feature encoding for regression head)
    def encode(self, x):
        # returns hidden activations (of shape (batch_size, H)) (each neuron encodes the input- return is the matrix of encodings of shape (N,H)- a row per sample and col per hidden neuron)
        return self.activation(self.hidden(x))
    
    #singular selected neuron affine transform and activation (neuron level encoding)
    def encode_single(self, x, neuron):

        w = self.hidden.weight[neuron].unsqueeze(0)
        b = self.hidden.bias[neuron]

        return self.activation(x @ w.T + b)

    
    #local reconstruction used in training (each neuron encodes and reconstructs its input)
    def reconstruct_single(self, x, neuron):

        #the latent representation for the neuron selected for perturbation (of size (batch_size,1)
        h_j = self.encode_single(x, neuron)

        # decoder weights leaving that neuron (since neurons in the decoder weights correspond to the cols) (of size (1,input_dim))
        w = self.decoder.weight[:, neuron].unsqueeze(0)   

        # full decoder bias (biases are shared across all neurons) (of shape (1,input_dim))
        b = self.decoder.bias.unsqueeze(0)      

        #reconstruction from this single neuron
        x_hat = self.activation(h_j * w + b)

        return x_hat
    
    #global reconstruction of the input (performed by the whole layer) used in plotting results
    def reconstruct_layer(self,x):

        #latent layer code (using pass through all hidden nodes)
        h = self.encode(x)

        #layer reconstructed input
        x_hat = self.activation(self.decoder(h))

        return x_hat

    #output layer task is regression (fitness values between [0,1]), thus return is sigmoid of affine of latent code
    def regress(self, x):

        h = self.encode(x)

        return self.activation(self.output(h))

