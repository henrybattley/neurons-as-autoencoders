
import numpy as np
import torch

class NKLandscape:

    """
    Implementation adapted from https://www.kaggle.com/code/elmiraaghdam/nk-model
    (Elmira Khodapanah Aghdam of Koc University Turkey on Kaggle (Kaggle.com))

    """

    def __init__(self, N: int, K: int, seed=None):
        self.N = N
        self.K = K

        #reproducible random generator from specified seed
        self.rng = np.random.default_rng(seed)

        #contribution_table defines the random landscape (the random values for each genes effect on fitness in all possible combinations with its interacting counterparts)
        self.contribution_table = self._generate_contribution_table()

        #defining the K interactions
        #we only need to compute interactions once (since interactions themselves don't change within a landscape)
        self.interaction_indices = []

        for i in range(self.N):

            interactions = []

            for j in range(self.K + 1): ##k+1 so every gene interacts with itself and k other genes 
                
                #using modulo operator such that we don't exceed the bound of the genome (N) (note that K must be an integer from 0 to N-1)
                index = (i + j) % self.N
                interactions.append(index)

            self.interaction_indices.append(interactions)

        """ interaction_indices may be written as list comprehension as follows, but I don't find it very readable
        self.interaction_indices = [
            [(i + j) % self.N for j in range(self.K + 1)] #k+1 so every gene interacts with itself
            for i in range(self.N)
        ]"""

        
    
    def _generate_contribution_table(self):
        num_combinations = 2 ** (self.K + 1)

        #returns matrix of shape (N, num_combinations), consisting of seeded random floats in range[0,1]
        return self.rng.random((self.N, num_combinations))
     


    def calculate_fitness(self, genome):
        fitness = 0
        #for each gene
        for i in range(self.N):
            
            #get the list of the genes they jointly contribute to fitness with (interact with) (ie. gene 0 may contribute to fitness of genome with [0,1,2] meaning itself and the values of the genome at index 1 & 2)
            indices = self.interaction_indices[i]
  
            #retrieve the value of those genes that interact with the current gene i in the sampled genome/structure
            interacting_states = [genome[j] for j in indices]
            #convert those binary integers to str, join each str, and interpret as a binary number
            index = int("".join(map(str, interacting_states)), 2)

            #add the fitness contribution of gene i at the contribution table position [i,index] to our sum
            #binary pattern of interacting genes is an address to the contribution table ie. 101==5, say 0th gene interacts with [0,1,2] indexes of the genome, as exemplified, and values are 101, retrieve 0th row 5th column from contribution_table
            fitness += self.contribution_table[i, index]

        #once all genes have their contributions summed we divide by N (number of genes parameterised)
        return fitness / self.N


    #generates a quantity of genomes and returns the set of genomes and their associated fitness values
    def sample(self,n_samples):
        #using our random seed for reproducabilty create the n_samples x N matrix of genomes (our X training set in regression terms)
        X = self.rng.integers(0, 2, size=(n_samples,self.N))
        #fitness values (target values in regression terms)
        y = []

        #for every genome in the sample size, calculate the fitness value and append that value to the y set
        for genome in X:

            fitness = self.calculate_fitness(genome)
            y.append(fitness)

        y = np.array(y)

        return X, y
    
    #send generated genomes and fitness to tensors for use with pytorch
    def get_dataset(N,K,n_samples,seed):

        print("Loading NK dataset...")

        landscape = NKLandscape(
            N=N,
            K=K,
            seed=seed
        )

        X, y = landscape.sample(n_samples)

        X = torch.tensor(X,dtype=torch.float32)

        #ensure that all 0 binary values are represented as -1s for networks (networks perform best when inputs are centered around zero with a balanced variance)
        X = 2 * X - 1

        y = torch.tensor(y,dtype=torch.float32).unsqueeze(1)

        NK_train = torch.utils.data.TensorDataset(X,y)


        return NK_train



