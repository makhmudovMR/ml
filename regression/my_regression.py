import numpy as np
import matplotlib.pyplot as plt
from mse import MSE
from statistics import mean



def get_a_b(xs,ys):
    b = sum((xs - mean(xs)) * (ys - mean(ys))) / sum((xs-mean(xs)) ** 2)
    a = mean(ys) - b * mean(xs)
    return a,b


def main():
    xs = np.array([1,2,3,4,5,6], dtype=np.float64)
    ys = np.array([2,2,4,3,6,4], dtype=np.float64)
    
    a,b = get_a_b(xs,ys)
    reg_line = [(b*x)+a for x in xs]
    
    plt.scatter(xs, ys)
    plt.plot(xs, reg_line)
    plt.show()

if __name__ == "__main__":
    main()
