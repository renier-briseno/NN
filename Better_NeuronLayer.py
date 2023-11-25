inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1],
             [0.5, -0.91, 0.26, -0.5],
             [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

outputs = []

for weight_list, bias in zip(weights, biases):
    neuron_output = 0

    for each_input, weight in zip(inputs, weight_list):
        neuron_output += each_input*weight

    neuron_output += bias
    outputs.append(neuron_output)

print(outputs)
