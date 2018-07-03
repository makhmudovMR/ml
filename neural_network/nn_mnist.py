import numpy as np
import pandas as pd
from sklearn import datasets

def nonline(x, deriv=False):
    if deriv:
        return x * (1-x)
    return 1 / (1 + np.exp(-x))

def main():
    np.random.seed(1)
    ds = datasets.load_digits()

    # print(len(ds.images), len(ds.target))
    X_train, y_train = ds.images[:1500], ds.target[:1500]
    X_test, y_test = ds.images[1500:], ds.target[1500:]
    # print(len(X_train), len(y_train))
    # print(len(X_test), len(y_test))
    # print(X_train.shape)
    # print(y_train[10:])

    wih = np.random.rand(32, 64)
    who = np.random.rand(10, 32)

    conv = [i.flatten() for i in X_train]
    X_train_vect = np.array(conv)
    # print(X_train_vect[:5])
    # print(np.squeeze(np.asarray(X_test[0])))

    # print(wih)
    # print(who)

    epochs = 5000
    for epoch in range(epochs):
        for x_t, y_t in zip(X_train_vect, y_train):
            print(x_t)
            hidden_output = nonline(np.dot(wih, x_t), False)
            final_output = nonline(np.dot(who, hidden_output))

            print(len(final_output))
            break
        break



if __name__ == '__main__':
    main()