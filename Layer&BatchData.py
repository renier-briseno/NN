import numpy as np

inputs = [[1, 2, 3, 2.5], 
          [2, 5, -1, 2], 
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Transpose weights array to create 3x4 * 4x3 matmul
outputs = np.dot(inputs, np.array(weights).T) + biases
print(outputs)