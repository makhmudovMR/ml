import numpy as np

class NN(object):

    def __init__(self, size):
        np.random.seed(1)
        self.size = size
        self.count_net = len(size)

        self.biases = [np.random.rand(y,1) for y in size[1:]]
        self.weights = [np.random.rand(y,x) for x,y in zip(size[:-1], size[1:])]
        self.sigmoid = lambda x: 1/(1+np.exp(-x))

    def train(self, x, y):
        pass

    def forward(self, x):
        a = x
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w,a)+b)
        return a


if __name__ == '__main__':
    nn = NN([2,3,4,1])
    for i in nn.weights:
        print(i, end='\n\n')

    print(nn.forward(np.array([1,1], ndmin=2).T))