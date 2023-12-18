class Neuron():
    def __init__(self, weight_list, bias):
        self.weights = weight_list
        self.bias = bias

    def evaluate(self, input_list):
        output = 0
        for input, weight in zip(input_list, self.weights):
            output += input*weight
        output += self.bias
        return output
    
# # Testing
# n1 = Neuron([1,2,3], 4)
# output = n1.evaluate([0.5, 0.5, 0.5])
# print(output)
