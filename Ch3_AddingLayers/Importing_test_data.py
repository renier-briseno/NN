import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from DenseLayerClass import Dense_Layer
from ReLU_Class import ReLU_Activation



# nnfs.init() method sets random seed to 0, creates a
# float32 dtype default, & overrides original
# dot product from numpy
nnfs.init() 

X, y = spiral_data(samples=100, classes=3)


dense1 = Dense_Layer(2, 3)
activation1 = ReLU_Activation()

dense1.forward(X)
activation1.forward(dense1.output)

print(dense1.output[:5])
print(activation1.output[:5])



# plt.scatter(X[:,0],X[:,1], c=y, cmap="brg")
# plt.show()