"""

@author: Lucas Pampolin Laheras
@project: PEL-208 - Task 3: kmeans implementation

"""

import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import plot


def dist(x1, x2, r):
    return sum(abs(x1 - x2) ** r) ** (1 / r)


def mean(data):
    return sum(data) / len(data)


# function to calculate the average of each characteristic of the whole sample
def grand_mean(data):
    return np.asarray([mean(data[i]) for i in range(len(data))])


# function to cluter using kmeans
def kmeans(data, k):
    C = np.asarray([np.random.uniform(np.min(data[:, i]), np.max(data[:, i]), size=k)
                    for i in range(data.shape[1])]).T

    olderC = np.zeros(C.shape)
    clusters = np.zeros(len(data))
    error = sum(dist(C, olderC, 2))

    while error != 0:

        for i in range(len(data)):
            distance = [dist(data[i], C[j], 2) for j in range(len(C))]
            clusters[i] = np.argmin(distance)

        olderC = C.copy()

        for i in range(k):
            points = np.asarray([data[j] for j in range(len(data)) if clusters[j] == i])
            if len(points) > 0:
                C[i] = grand_mean(points.T)

        if data.shape[1] == 2:
            plot.plotClusters(data, clusters, C)

        error = sum(dist(C, olderC, 2))

    return clusters


files = ["data_task03/wine.csv", "data_task03/divorce.csv"]

iris = datasets.load_iris()
names = iris.target_names
data = iris.data
labels = iris.target

clusters = kmeans(data, 3)
print(clusters)

for i in range(3):
    print(names[i])
    print(sum(clusters == i))
    print(sum(labels == i))
    print(min(sum(clusters == i), sum(labels == i)) / max(sum(clusters == i), sum(labels == i)) * 100, "%")


for fname in files:
    data = pd.read_csv(fname, sep=';', header=None)
    data.dropna(inplace=True)
    data = data.to_numpy()

    labels = pd.read_csv(fname[:len(fname) - 3] + "target")
    labels = labels.to_numpy()

    classes = np.unique(labels)
    k = len(classes)

    clusters = kmeans(data, k)
    print("Have", clusters, "clusters")
    for i in range(k):
        print(classes[i])
        print(sum(clusters == i))
        print(sum(labels == classes[i]))
        print(min(sum(clusters == i), sum(labels == classes[i])) / max(sum(clusters == i),
                                                                       sum(labels == classes[i])) * 100, "%")
