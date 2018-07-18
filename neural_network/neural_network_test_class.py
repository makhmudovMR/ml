import numpy as np

class NN(object):

    def __init__(self, size):
        np.random.seed(1)
        self.size = size
        self.count_layer = len(size)

        self.biases = [np.random.rand(y,1) for y in size[1:]]
        self.weights = [np.random.rand(y,x) for x,y in zip(size[:-1], size[1:])]
        self.sigmoid = lambda x: 1/(1+np.exp(-x))

        self.activation = []
        self.lr = 0.01

    def train(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # print(nabla_w)
        # print(nabla_b)
        a = x
        activations = [x]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,a)+b
            a = self.sigmoid(z)
            activations.append(a)
            zs.append(z)

        delta = (activations[-1] - y) * self._sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.count_layer):
            z = zs[-l]
            sp = self._sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        self.weights = [w - self.lr * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - self.lr * nb for b, nb in zip(self.biases, nabla_b)]



    def forward(self, x):
        a = x
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w,a)+b)
        return a

    def _sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


if __name__ == '__main__':
    nn = NN([2,5,2])
    x = np.array([0,1], ndmin=2).T
    y = np.array([1,0], ndmin=2).T
    print(nn.forward(x))
    for i in range(2000):
        nn.train(x,y)
    print(nn.forward(x))

