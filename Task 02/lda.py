import numpy as np
import matplotlib.pyplot as plt

# def LDA(data, labels):
#     newData = np.asarray([data[:, labels == i] for i in range(max(labels) + 1)])
#
#     grandMean = np.asarray([np.mean(data[i]) for i in range(len(data))])
#     grandMean.reshape(grandMean.shape[0], 1)
#
#     classMeans = np.asarray([np.asarray([np.mean(data[i]) for i in range(len(newData[i]))]) for i in range(len(newData))])
#
#     Sw = np.zeros((data.shape[1], data.shape[1]))
#     for i in range(len(data)):
#         Si = np.zeros((data.shape[1], data.shape[1]))
#         for j in range(data.shape[2]):
#             row, mv = data[i][:, j].reshape(data.shape[1], 1), classMeans[i].reshape(data.shape[1], 1)
#             Si += (row - mv).dot((row - mv).T)
#         Sw += Si
#
#     Sb = np.zeros((data.shape[1], data.shape[1]))
#     for i in range(len(classMeans)):
#         meanV, meanG = classMeans[i].reshape(data.shape[1], 1), grandMean.reshape(data.shape[1], 1)
#         Sb += data[i].shape[1] * (meanV - meanG).dot((meanV - meanG).T)
#
#     eigVal, eigVec = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
#
#     eigPair = [(np.abs(eigVal[i]), eigVec[:, i]) for i in range(len(eigVal))]
#     eigPair = sorted(eigPair, key=lambda k: k[0], reverse=True)
#
#
#     W = []
#     for i in range(k):
#         W.append(eigPair[i][1].reshape(len(eigPair[i][1]), 1))
#     W = np.asarray(W)
#     data_lda = data.T.dot(W)
#     return data_lda


class LDA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.eig_vectors = None

    def transform(self, X, y):
        height, width = X.shape
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        scatter_t = np.cov(X.T) * (height - 1)
        scatter_w = 0
        for i in range(num_classes):
            class_items = np.flatnonzero(y == unique_classes[i])
            scatter_w = scatter_w + np.cov(X[class_items].T) * (len(class_items) - 1)

        scatter_b = scatter_t - scatter_w
        _, eig_vectors = np.linalg.eigh(np.linalg.pinv(scatter_w).dot(scatter_b))
        print(eig_vectors.shape)
        pc = X.dot(eig_vectors[:, ::-1][:, :self.n_components])
        print(pc.shape)

        if self.n_components == 2:
            if y is None:
                plt.scatter(pc[:, 0], pc[:, 1])
            else:
                colors = ['r', 'g', 'b']
                labels = np.unique(y)
                for color, label in zip(colors, labels):
                    class_data = pc[np.flatnonzero(y == label)]
                    plt.scatter(class_data[:, 0], class_data[:, 1], c=color)
            plt.show()
        return pc