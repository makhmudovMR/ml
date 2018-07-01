# we will use iris dataset
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import numpy as np

# load the dataset
data = load_iris()

model = GaussianNB()
model.fit(data.data, data.target)

# evalaute
print(model.score(data.data, data.target))
# output = 0.96

# predict
model.predict([4.2, 3, 0.9, 2.1])
# 0 = setosa,1 = versicolor, and 2 = virginica