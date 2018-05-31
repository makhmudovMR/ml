import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors

def main():
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1 , inplace=True)

    x = np.array(df.drop['class'],1)
    y = np.array(df['class'])

    X_train, X_train, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train,)

if __name__ == "__main__":
    main()
