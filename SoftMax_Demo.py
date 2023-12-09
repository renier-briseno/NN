import math 

outputs = [4.8, 1.21, 2.385]
exp_values = []

for output in outputs:
    exp_values.append(math.e ** output)

print("exponentiated values:")
print(exp_values)
