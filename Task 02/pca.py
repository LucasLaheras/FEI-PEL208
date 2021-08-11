import numpy as np

# import pandas
# import matplotlib.pyplot as plt
# from pca import PCA
# from lda import LDA
# import numpy as np
#
# FILE_DATA_FISHER = 'data_task02/iris_data_set.csv'
#
# data = pandas.read_csv(FILE_DATA_FISHER, sep=',')
# matrix = data.values.tolist()
#
# x = [row[1:-1] for row in matrix]
# y = [[row[-1] for row in matrix]]
#
#
# pca_value = PCA(x, 0.98)
# print(pca_value)

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# data = pandas.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
#
# # prepare the data
# x = data.iloc[:, 0:4]
# target = data.iloc[:, 4]
#
# # Executing PCA
# mat_reduced1 = PCA(x, 1)
# mat_reduced2 = PCA(x, 2)
# mat_reduced3 = PCA(x, 3)
#
# # Executing LDA
# sections = LDA(n_components=2)
# transformed = sections.transform(x, target)
# print(transformed)
#
# # Transforming in dataframe
# matrix_df1 = pandas.DataFrame(mat_reduced1, columns=['PC1'])
# matrix_df2 = pandas.DataFrame(mat_reduced2, columns=['PC1', 'PC2'])
# matrix_df3 = pandas.DataFrame(mat_reduced3, columns=['PC1', 'PC2', 'PC3'])
#
#
# # set color target
# colors = target.copy()
# classes = list(set(colors))
# for i in range(len(colors)):
#     if colors[i] == classes[0]:
#         colors[i] = 'r'
#     elif colors[i] == classes[1]:
#         colors[i] = 'g'
#     elif colors[i] == classes[2]:
#         colors[i] = 'b'
#
# # plot
# plt.scatter(matrix_df1['PC1'], np.zeros_like(matrix_df1['PC1']), c=colors)
# plt.show()
#
# plt.scatter(matrix_df2['PC1'], matrix_df2['PC2'], c=colors)
# plt.show()
#
# ax = plt.axes(projection='3d')
# ax.scatter3D(matrix_df3['PC1'], matrix_df3['PC2'], matrix_df3['PC3'], c=colors)
# plt.show()



def PCA(X, num_components):
    # subtract the mean
    matrix_mean = X - np.mean(X, axis=0)

    # calculate the covariance matrix
    cov_mat = np.cov(matrix_mean, rowvar=False)

    # calculate eigenvectors and eigenvalues of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # choose components and form a feature vector
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # new data
    data_reduced = np.dot(eigenvector_subset.transpose(), matrix_mean.transpose()).transpose()

    return data_reduced

