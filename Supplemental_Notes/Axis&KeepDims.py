import numpy as np

outputs = np.array([[4.8, 1.21, 2.385],
                    [8.9, -1.81, 0.2],
                    [1.41, 1.051, 0.026]])

print("Sum without axis")
print(np.sum(outputs))

print("Sum with axis=None")
print(np.sum(outputs, axis=None))
# We are essentially just summing all the elements

print("Sum with axis=0") # "row-wise" is essentially summing columns
axis0 = np.sum(outputs, axis=0)
print(axis0)
print(np.shape(axis0))

print("Sum with axis=1") # "column-wise" is essentially summing rows
axis1 = np.sum(outputs, axis=1)
print(axis1)
print(np.shape(axis1))

"""
    BOTH GIVE A (3,) SHAPE
"""

# By keeping dims, we have a (3,1) shape
print("Sum with axis=1, keepdims=True") # "column-wise" is essentially summing rows
axis1 = np.sum(outputs, axis=1, keepdims=True)
print(axis1)
print(np.shape(axis1))