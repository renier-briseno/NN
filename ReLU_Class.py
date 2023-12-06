import numpy as np

class ReLU_Activation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)