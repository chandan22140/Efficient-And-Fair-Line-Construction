import torch
import numpy as np
from sklearn.datasets import fetch_california_housing


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


def gradient_descent(point_set, alpha=0.1, max_iterations=1000):
    a = torch.rand(2, requires_grad=True)
    a.data = a.data / torch.norm(a.data)
    b = torch.tensor([np.mean(point_set[:, 0]), np.mean(point_set[:, 1])], requires_grad=True)

    point_set = torch.tensor(point_set, dtype=torch.float)

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

        if i % 100 == 0:
            max_distance = cost.item()
            print(f"Iteration {i}: Maximum distance = {max_distance:.4f}")

    return a.detach().numpy(), b.detach().numpy()

def dataset():
    k = fetch_california_housing(as_frame=True)
    data = k.frame[['Longitude', 'Latitude']]
    return data.values

point_set = dataset() 
fair_line = gradient_descent(point_set)
