import math 
import numpy as np


outputs = [4.8, 1.21, 2.385]

"""
            using math
"""
# exp_values = []
# for output in outputs:
#     exp_values.append(math.e ** output)

# # print("exponentiated values:")
# # print(exp_values)

# norm_base = sum(exp_values)
# norm_values = []
# for value in exp_values:
#     norm_values.append(value/norm_base)
# print("Normalized exponentiated values:")
# print(norm_values)

# print("Sum of normed values:", sum(norm_values))

#                      OR

"""
        using Nunmpy
"""

# exp_values = np.exp(outputs)
# print("exponentiated values:")
# print(exp_values)

# norm_values = exp_values / np.sum(exp_values)
# print("normalized exponentiated values:")
# print(norm_values)
# print("sum of normalized values:", np.sum(norm_values))


#        OR IN ONE STEP

exp_values = np.exp(outputs) #"outputs" are the layers inputs

# Axis: 0=rows, 1=columns
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
