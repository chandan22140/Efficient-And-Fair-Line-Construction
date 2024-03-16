import numpy as np
import torch
from sklearn.datasets import fetch_california_housing

cost_fnal = {} # GLOBAL

def dataset():
    k = fetch_california_housing(as_frame=True)
    data = k.frame[['Longitude', 'Latitude']]
    return data.values

def distance_to_line(p, a, b):
    identity = np.identity(len(a))
    return np.linalg.norm((identity - np.outer(a, a)).dot(p - b)) ** 2

def cost_function(point_set, a, b):
    point_set = torch.tensor(point_set, dtype=torch.float)
    a = torch.tensor(a, dtype=torch.float, requires_grad=True)
    b = torch.tensor(b, dtype=torch.float, requires_grad=True)

    distances = torch.norm((torch.eye(2) - torch.outer(a, a)).matmul((point_set - b).T), dim=0)
    max_distance = torch.max(distances)
    return max_distance

def gradient_descent(point_set, alpha=0.1, max_iterations=10):
    a = torch.rand(2, requires_grad=True)
    a.data = a.data / torch.norm(a.data)
    b = torch.tensor([np.mean(point_set[:, 0]), np.mean(point_set[:, 1])], requires_grad=True)
    point_set = torch.tensor(point_set, dtype=torch.float)  # Convert point_set to tensor once
    
    optimizer = torch.optim.Adam([a, b], lr=alpha)
    
    for i in range(max_iterations):
        # Zero the gradients
        optimizer.zero_grad()

        # Compute the cost function
        cost = cost_function(point_set, a, b)

        # Backpropagation
        cost.backward()

        # Update parameters
        optimizer.step()
        max_distance = cost.item()
        print(f"Iteration {i}: Maximum distance = {max_distance:.4f}")

    cost_fnal[(tuple(a.detach().numpy()), tuple(b.detach().numpy()))] = cost.item() # Append the scalar value of cost after all iterations
    
    return a.detach().numpy(), b.detach().numpy()

point_set = dataset()[:10]

for i in range(1000):
    fair_line = gradient_descent(point_set)

# Find the minimum cost and its corresponding fair line parameters
min_cost = min(cost_fnal.values())
min_cost_params = [k for k, v in cost_fnal.items() if v == min_cost][0]

print("Minimum cost:", min_cost)
print("Corresponding fair line parameters:", min_cost_params)
