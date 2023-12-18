import numpy as np

class Softmax():
    def forward(self, inputs):
        # We delete large values because exponentiation will make them unimaginably greater
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        pdf = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = pdf


# softmax = Softmax()
# softmax.forward([[1, 2, 3]])
# print(softmax.output)

# softmax.forward([[-2, -1, 0]])
# print(softmax.output)
"""
    *As you can see, minusing 3 (the max) from the first array,
    and minusing 0 (the max) from the second array yields the same array,
    Therefore, the softmax.forward() method will return the same output
"""

"""
    Dividing [1, 2, 3] by two yields [0.5, 1, 1.5], After subtraction
    of the max, we get [-1, -0.5, 0], which would be the same outputs of
    [-2, -1, 0] / 2. However, this changes the confidences (the pdf), this
    will be later explained with the concept of scaling & nonlinearity
"""
