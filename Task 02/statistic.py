import numpy as np


def mean(data):
    return sum(data)/len(data)


def grand_mean(data):
    return np.asarray([mean(data[i]) for i in range(len(data))])


def within_class(data, classMeans):
    Sw = np.zeros((data.shape[1],data.shape[1]))
    for i in range(len(data)):
        Si = np.zeros((data.shape[1],data.shape[1]))
        for j in range(data.shape[2]):
            row, mv = data[i][:,j].reshape(data.shape[1], 1), classMeans[i].reshape(data.shape[1], 1)
            Si += (row - mv).dot((row-mv).T)
        Sw += Si
    return Sw

# calculate matrix between-class
def between_class(grandMean, classMeans, data):
    Sb = np.zeros((data.shape[1], data.shape[1]))
    for i in range(len(classMeans)):
        meanV, meanG = classMeans[i].reshape(data.shape[1], 1), grandMean.reshape(data.shape[1], 1)    
        Sb += data[i].shape[1] * (meanV - meanG).dot((meanV - meanG).T)
    return Sb

# LDA calc
def LDA(data, labels):
    newData = np.asarray([data[:,labels == i] for i in range(max(labels)+1)])
    
    grandMean = grand_mean(data)
    grandMean.reshape(grandMean.shape[0], 1)
    
    classMeans = np.asarray([grand_mean(newData[i]) for i in range(len(newData))])
    
    Sw = within_class(newData, classMeans)
    Sb = between_class(grandMean, classMeans, newData)
    
    eigVal, eigVec = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    
    
    eigPair = [(np.abs(eigVal[i]), eigVec[:,i]) for i in range(len(eigVal))]
    eigPair = sorted(eigPair, key = lambda k: k[0], reverse = True)
    
    return eigPair


# transform data to new coordinates
def LDA_transform(data, eigPair, k):
    W = []
    for i in range(k):
        W.append(eigPair[i][1].reshape(len(eigPair[i][1]), 1))
    W = np.asarray(W)
    data_lda = data.T.dot(W)
    return data_lda


# Covariance between two variables
def covariance(x, y):
    
    a = [i - mean(x) for i in x]
    b = [i - mean(y) for i in y]
 
    c = [a[i]*b[i] for i in range(len(a))]
    
    return sum(c)/(len(x)-1)


# Covariance between matrix
def getCovMatrix(data):
    cov = [[covariance(data[x],data[y]) for x in range(len(data))] for y in range(len(data))]
    return cov
 

# PCA calc
def PCA(data):
    covMatrix = getCovMatrix(data)
    eigenValues, eigenVector = np.linalg.eig(covMatrix)
    eigPair = [(np.abs(eigenValues[i]), eigenVector[:,i]) for i in range(len(eigenValues))]
    eigPair = sorted(eigPair, key = lambda k: k[0], reverse = True)
    
    return eigPair

# transform data to new coordinates
def PCA_transform(data, eigPair, k):
    W = []
    for i in range(k):
        W.append(eigPair[i][1].reshape(len(eigPair[i][1]), 1))
    W = np.asarray(W)
    data_pca = data.T.dot(W)
    return data_pca


def PCA_inverse_transform(data, eigVec, mean):
    mult = np.matmul(np.linalg.inv(eigVec.T), data)
    for i in range(len(mult)):
        for j in range(len(mult[i])):
            mult[i][j] + mean[i]
    return mult

