import numpy as np

sigmoid = lambda x : 1/(1+np.exp(-x))

def perceptron(m1,m2,w1,w2,b):
    res = m1 * w1 + m2 * w2 + b
    return sigmoid(res)

print(perceptron(1,1,0.6, 0.3, 0))
