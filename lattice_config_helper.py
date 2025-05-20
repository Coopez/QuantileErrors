# def ceil(a, b):
#     return -(a // -b)
# input_size = 512
# quantile_size = 1
# input_per_lattice_per_layer = [3, 3, 3, 3, 3, 3]
# output_size = 90
# keypoints = 10

# complexity = [ceil((input_size + quantile_size),input_per_lattice) * keypoints ** (input_per_lattice) for input_per_lattice in input_per_lattice_per_layer]
# total_complexity = sum(complexity)

# print(total_complexity)
# print(complexity)




import matplotlib.pyplot as plt
import numpy as np


def return_complexity(n_features, n_keypoints, number):
    return number * (n_keypoints ** n_features)


def choose_2(n_features):
    return n_features * (n_features - 1) / 2

def choose_3(n_features):
    return n_features * (n_features - 1) * (n_features - 2) / 6

def choose_4(n_features):
    return n_features * (n_features - 1) * (n_features - 2) * (n_features - 3) / 24

feature_scale = np.arange(2, 20, 1)
keypoints_scale = 2 #np.arange(2, 20, 1)
complexity_1 = return_complexity(feature_scale, keypoints_scale,1)
complexity_2 = return_complexity(2, keypoints_scale, choose_2(feature_scale))
complexity_3 = return_complexity(2, keypoints_scale, choose_3(feature_scale))
complexity_4 = return_complexity(2, keypoints_scale, choose_4(feature_scale))

plt.plot(feature_scale, complexity_1, label='normal lattice')
plt.plot(feature_scale, complexity_2, label='choose 2 features')
plt.plot(feature_scale, complexity_3, label='choose 3 features')
plt.plot(feature_scale, complexity_4, label='choose 4 features')
plt.legend()
plt.xlabel('Number of features')
plt.ylabel('Complexity')
plt.title('Complexity of Lattice Model')
plt.show()


