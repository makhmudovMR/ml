import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')


dataset = {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}

new_features = [5,7]

for i in dataset:
    print(dataset[i])
    for ii in dataset[i]:
        print(ii)
