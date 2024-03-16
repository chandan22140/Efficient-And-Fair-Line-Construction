

import numpy as np
from sklearn.datasets import fetch_california_housing


# Function to load dataset
def dataset():
    k = fetch_california_housing(as_frame=True)
    data = k.frame[['Longitude', 'Latitude']]
    return data.values

# Function to compute distance from a point to a line
def distance_to_line(p, a, b):
    identity = np.identity(len(a))
    return np.linalg.norm((identity - np.outer(a, a)).dot(p - b))**2

# Function to compute total cost
def total_cost(points, lines):
    total = 0
    for p in points:
        min_distance = float('inf')
        for line in lines:
            distance = distance_to_line(p, line[0], line[1])
            min_distance = min(min_distance, distance)
        total += min_distance
    return total

# Function to compute gradients of cost function for multiple lines
def compute_gradients_multiple(points, lines):
    num_lines = len(lines)
    grad_a = np.zeros((num_lines, 2))
    grad_b = np.zeros((num_lines, 2))
    for p in points:
        for i, line in enumerate(lines):
            a, b = line
            V = p - b
            Vt = V.T
            outer_prod = np.outer(a, a)
            term1 = (np.identity(len(a)) - outer_prod).dot(V)
            grad_a[i] += -2 * (Vt.dot(a)) * V + (Vt.dot(a)) * (a.T.dot(a)) * V + (Vt.dot(a)) * (a.T.dot(V)) * a
            grad_b[i] += -V + 2 * (Vt.dot(a)) * a - (a.T.dot(a)) * (Vt.dot(a)) * a
    return grad_a, grad_b

# Gradient descent optimization for multiple lines
# def gradient_descent_multiple(points, lines_init, learning_rate=0.001, max_iters=100, tolerance=1e-6):
#     lines = lines_init.copy()
#     for _ in range(max_iters):
#         cost = total_cost(points, lines)
#         grad_a, grad_b = compute_gradients_multiple(points, lines)
#         grad_a = np.nan_to_num(grad_a)
#         grad_b = np.nan_to_num(grad_b)
#         lines -= learning_rate * grad_a, learning_rate * grad_b
#         new_cost = total_cost(points, lines)
#         if np.abs(cost - new_cost) < tolerance:
#             break
#     return lines, new_cost


def gradient_descent_multiple(points, lines_init, learning_rate=0.001, max_iters=100, tolerance=1e-6):
    lines = lines_init.copy()
    cost = total_cost(points, lines)
    for _ in range(max_iters):
        grad_a, grad_b = compute_gradients_multiple(points, lines)
        grad_a = np.nan_to_num(grad_a)
        grad_b = np.nan_to_num(grad_b)
        
        # Update lines individually
        for i in range(len(lines)):
            lines[i, :, 0] -= learning_rate * grad_a[i]
            lines[i, :, 1] -= learning_rate * grad_b[i]
            # norm_a = np.linalg.norm(lines[i, :, 0])
            # # norm_b = np.linalg.norm(lines[i, :, 1])
            # if norm_a>1e-6:
            #     # lines[i, :, 0] /= norm_a
            #     o = []
            #     # lines[i, :, 1] /= norm_b
            # print(lines[i, :, 0])
            # print(lines[i, :, 1])
            
        new_cost = total_cost(points, lines)
        if np.abs(cost - new_cost) < tolerance:
            # break
            cost = new_cost
    return lines, cost

# Function to initialize lines randomly
def initialize_lines(num_lines, points):
    lines_init = []
    for _ in range(num_lines):
        # Initialize direction vector randomly
        a_init = np.random.rand(2)
        a_init /= np.linalg.norm(a_init)  # Ensure unit norm
        
        # Initialize point on the line by averaging all points
        b_init = np.mean(points, axis=0)
        lines_init.append((a_init, b_init))

    return np.array(lines_init)

# Example usage
points = dataset()[0:500]
num_lines = int(input("Please enter number of lines"))
lines_init = initialize_lines(num_lines, points)
lines_opt, min_cost = gradient_descent_multiple(points, lines_init)
print("Optimal lines:")
for i, line in enumerate(lines_opt):
    print("Line", i+1)
    print("Direction vector a:", line[0])
    print("Point b on the line:", line[1])
print("Minimum cost:", min_cost)
