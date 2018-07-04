import numpy as np

class Perceptron(object):

    def __init__(self, x, y, b):
        self.w = np.zeros((1, len(x)))
        self.x = np.array(x, ndmin=2).T
        self.y = y
        self.b = b
        self.lr = 0.01
        # print(self.w)
        # print(self.w.shape)

    def predict(self):
        return np.dot(self.w, self.x)

    def train(self):
        pred = self.predict()
        error = (pred - self.y)**2
        self.w += -self.lr * (pred - self.y)


def main():
    p = Perceptron([1, 0.2, 0.8], 1, 0)
    print(p.predict())
    print(p.w)
    epochs = 4000
    for i in range(epochs):
        p.train()
    print(p.predict())
    print(p.w)


if __name__ == '__main__':
    main()