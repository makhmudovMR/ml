import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')


dataset = {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]} # object group
new_features = [5,7] # new object for predict

##[[plt.scatter(ii[0], ii[1],s=100, color=i) for ii in dataset[i]] for i in dataset]
##plt.scatter(new_features[0], new_features[1],s=100, color='y')
##plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')
    distances = []
    for group in data:
        print('group {}:'.format(group))
        for features in data[group]:
            print("   features {}".format(features))
            #euclidean_distance = sqrt((fratures[0] - predict[0])**2 - (features[1] - predict[1])**2))
            euclidean_distance = np.linalg.norm(np.array(features)- np.array(predict))
            distances.append([euclidean_distance, group])
    
    print("\ndistances : {}".format(distances))
    
    votes = [i[1] for i in sorted(distances)[:k]]
    
    print("\nvotes {}".format(votes))
    print("\n",Counter(votes).most_common(1))
    
    vote_result = Counter(votes).most_common(1)[0][0]
    
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

[[plt.scatter(ii[0], ii[1],s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1],s=100, color=result)
plt.show()
