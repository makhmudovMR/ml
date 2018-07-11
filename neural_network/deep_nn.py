import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Network(object):

    def __init__(self, arch):
        self.num_layer = len(arch)
        self.arch = arch
        self.weight = [np.random.randn(y,x) for x,y in zip(arch[:-1], arch[1:])]


    def feedforward(self, a):
        for w in self.weight:
            a = sigmoid(np.dot(w,a))
        return a


if __name__ == '__main__':
    nn = Network([2, 3, 2])
    print(nn.weight)
    print(nn.feedforward(np.array([1,1], ndmin=2).T))