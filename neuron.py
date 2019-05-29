import numpy as np
def sigmoid(x):
    return (1/(1+np.exp(-x)))
class Neuron:
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias
    def feedforward(self,inputs):
        total = np.dot(self.weights,inputs) + bias
        return sigmoid(total)
weights = np.array([0,1,2])
bias = 5
input = np.array([2,3,8])

class MyNetwork:
    def __init__(self):
        weights = np.array([0,1,2])
        bias = 2
        self.h1 = Neuron(weights,bias)
        self.h2 = Neuron(weights,bias)
        self.h3 = Neuron(weights,bias)
        self.o1 = Neuron(weights,bias)
    def feedforward(self,x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_h3 = self.h3.feedforward(x)
        out_o1 = self.o1.feedforward([out_h1,out_h2,out_h3])
        return out_o1
a = MyNetwork()
print(a.feedforward(input))
