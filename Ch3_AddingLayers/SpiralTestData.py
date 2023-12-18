import sys
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from ..Ch3_AddingLayers.DenseLayerClass import Dense_Layer
from ..Ch4_ActivationFunctions.ReLU_Class import ReLU
from ..Ch4_ActivationFunctions.SoftMax_Class import Softmax


"""
    Please run this modeule as a script from one directory above "NN".

    Example terminal command from desktop:
        python -m NN.Ch3_AddingLayers.SpiralTestData 
"""


# nnfs.init() method sets random seed to 0, creates a
# float32 dtype default, & overrides original
# dot product from numpy
nnfs.init() 

X, y = spiral_data(samples=100, classes=3)


dense1 = Dense_Layer(2, 3)
activation1 = ReLU()

dense2 = Dense_Layer(3, 3)
activation2 = Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])




# Previous code to see forward interaction with outputs
"""
dense1.forward(X)
activation1.forward(dense1.output)

print(dense1.output[:5])
print(activation1.output[:5])
"""


# Code to print the Spiral data out
# plt.scatter(X[:,0],X[:,1], c=y, cmap="brg")
# plt.show()