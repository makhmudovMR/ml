import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Network(object):

    def __init__(self, arch):
        self.num_layer = len(arch)
        self.arch = arch
        self.weight = [np.random.randn(y,x) for x,y in zip(arch[:-1], arch[1:])]

        self.activations = []
        self.layer_error = []


    def feedforward(self, a):
        for w in self.weight:
            a = sigmoid(np.dot(w,a))
            self.activations.append(a)
        return a

    def train(self, a, y):
        self.activations.clear()
        self.feedforward(a)

        output_error = y - self.activations[-1]
        error = output_error
        for w in reversed(self.weight):
            error = np.dot(w.T, error)
            self.layer_error.append(error)
        self.layer_error.reverse()
        self.layer_error.append(output_error)

        


if __name__ == '__main__':
    nn = Network([2, 2, 1])
    for a in nn.activations:
        print(a)

    x = np.array([1,1], ndmin=2).T
    y = np.array([1], ndmin=2).T
    nn.train(x,y)
    for i in nn.layer_error:
        print(i)