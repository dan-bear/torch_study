import torch


def test():
    num_of_data_points = 32
    data_set = [torch.empty(8, dtype=torch.uint8).random_(0,2)\
                for i in range(num_of_data_points)]
    labels_set = [torch.sum(vector).item() for vector in data_set]
    for idx in range(num_of_data_points):
        print(f"idx = {idx}, vec = {data_set[idx]}, label = {labels_set[idx]}")

if __name__ == "__main__":
    test()

# import torch
# import torch.nn as nn
# import torch.optim as optim

# class CountOnesNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(CountOnesNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = torch.sigmoid(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Define input vector
# input_vector = torch.tensor([[0, 1, 0, 1], [1, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.float32)

# # Define labels (number of 1s in each vector)
# labels = torch.tensor([[2], [3], [2]], dtype=torch.float32)

# # Define hyperparameters
# input_size = 4  # size of input vector
# hidden_size = 8  # size of hidden layer
# output_size = 1  # size of output (number of 1s)

# # Create the neural network
# model = CountOnesNN(input_size, hidden_size, output_size)

# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# # Training loop
# num_epochs = 1000
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(input_vector)
#     # Calculate loss
#     loss = criterion(outputs, labels)
#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Test the model
# test_input = torch.tensor([[1, 0, 1, 1]], dtype=torch.float32)
# with torch.no_grad():
#     output = model(test_input)
#     print(f'Input: {test_input}, Output: {output.item()}')
