# Let's build a NN which gets a vector in {0,1}^16 and 
# returns the number of 1's in the vector, in other words
# it sums the vector's entry.

import torch

# torch.nn namespace provides the building blocks to build
# a neural network
import torch.nn as nn 

# torch.optim namespace implements various optimizations
# algorithms.
import torch.optim as optim


#__________________________________________________________#
#HYPER PARAMETERS:
# Each vector is in {0,1}^16, so the input dimension is 16.
input_vector_dim = 16

# Size of the hidden layer.
# todo 240225 - why 32 should be used? what happens if
#               smaller size is used? and bigger size?
hidden_layer_size = 32

# The output is a single integer (a 1x1 vector).
output_vec_dim = 1

#__________________________________________________________#
#DATA SET
# todo 240225 - for now the data set and the lables are 
#               stored in a list, but one could have used
#               another torch tensor and iterate it, maybe 
#               it's a better approach?

# Number of data points
# todo 240225: why 32? maybe more are needed? maybe less? 
num_of_data_points = 32

# Data list
data_set = []

# Add data points:
# todo 240225: since random is used, there might be a vector
#              that appears twice in the data set, it can be
#              cleaned, but for now I decided not to do it
#              since I feel luck.
# todo 240225: for now the list holds tensors and not lists
#              Tensor.tolist() allows to extract the 
#              tensor value to a list.
#
# Notes:
# 1: torch.empty(<dim>, dtyp=<type>) creates a 1 X dim
#    tensor without initializing its memory. The 
#    returned value is a torch.Tensor object.
# 2: torch.Tensor.random_(0, 2) sets each of the
#    tensor's entries to a random integer value in range
#    [0, 2) (so it can be 0 or 1 in this case).

data_set = [torch.empty(input_vector_dim, dtype=torch.uint8).random_(0,2)\
            for dataIdx in range(num_of_data_points)]
        
# Torch provides sum method that sums the values of a
# tensor and returns a tensor:
# torch.sum(<Tensor object>) -> Tensor. In our case it's a
# 1X1 tensor. torch.Tensor provides the item() method that
# extract the value in a 1X1 tensor (tolist should be used
# if the tensor is multi-dimensional). I decided to keep the
# labels as simple python integers and not tensors.
# 
# todo 240225 - should one just used tensors and avoid
#               python lists altogether?
labels_set = [torch.sum(vector).item() for vector in data_set]


#__________________________________________________________#







# The count-ones-nn derives from nn.Module, the nn.Module
# is a class defined in the torch.nn namespace and all the
# NN implemented using torch should derive from nn.Module
class CountOnesNN(nn.Module):
    # Initializing the class:
    # 1: Must call the nn.Module __init__ method.
    # 2: The NN is going to contain only 1 hidden layer
    # 3:
    # 4:
    # 5: 
    def __init__(self, input_size, hidden_layer_size, output_size)




if __name__ == "__main__":
    main()
