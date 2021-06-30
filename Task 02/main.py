"""

@author: Lucas Pampolin Laheras
@project: PEL-208 - Task 2: Principal Component Analysis (PCA) and Latent Dirichlet Allocation (LDA) implementation

"""

import statistic
import plot
from sklearn import datasets
from sklearn.decomposition import PCA

# read database
iris = datasets.load_iris()
names = iris.target_names
data = iris.data
labels = iris.target
data = data.T

# Applying LDA
lda = statistic.LDA(data, labels)
# Applying PCA
pca = statistic.PCA(data)

# transform data using 1 eigenvector
lda_data = statistic.LDA_transform(data, lda, 1)
pca_data = statistic.PCA_transform(data, pca, 1)

# Plot 1D
plot.transformedData1D(lda_data, labels, '', names)
plot.transformedData1D(pca_data, labels, '', names)


# transform data using 2 eigenvectors
lda_data = statistic.LDA_transform(data, lda, 2)
pca_data = statistic.PCA_transform(data, pca, 2)

# Plot 2D
plot.transformedData2D(lda_data, labels, '', names)
plot.transformedData2D(pca_data, labels, '', names)


for i in range(1, 4):
    pca = PCA(n_components=i)
    pca.fit(data.T)
    pca_data = pca.transform(data.T)

    lda = statistic.LDA(pca_data.T, labels)
    if i == 1:
        lda_data = statistic.LDA_transform(pca_data.T, lda, 1)
        plot.transformedData1D(lda_data, labels, '', names)
    else:
        lda_data = statistic.LDA_transform(pca_data.T, lda, 2)
        plot.transformedData2D(lda_data, labels, '', names)



