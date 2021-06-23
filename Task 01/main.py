"""

@author: Lucas Pampolin Laheras
@project: PEL-208 - Task 1: Linear Regression by Least Squares Method

"""

import pandas
from linear_regression import LinearRegression
import matplotlib.pyplot as plt
from matrix import *
import numpy as np


FILE_DATA_CENSUS = 'Task 01/data_task01/US-Census.csv'
FILE_DATA_BOOKS = 'Task 01/data_task01/Books_attend_grade.csv'
FILE_DATA_ALPSWATER = 'Task 01/data_task01/alpswater.csv'


if __name__ == '__main__':
    # US Census dataset
    data = pandas.read_csv(FILE_DATA_CENSUS, sep=';')
    data = data.values.tolist()

    x = [row[0:-1] for row in data]

    y = [[int(row[-1].replace('.', '')) for row in data]]

    lin_re = LinearRegression(x, y)

    ans = lin_re.linear_least_squares()
    print(ans)
    ans1 = lin_re.quadratic_least_squares()
    print(ans1)

    a = np.arange(1890, 2010, 1)

    plt.scatter(x, y)
    plt.plot(a, a*ans[1][0] + ans[0][0], 'g')
    plt.plot(a, a*a*ans1[2][0] + a*ans1[1][0] + ans1[0][0], 'r')
    plt.show()


    # alpswater dataset
    data = pandas.read_csv(FILE_DATA_ALPSWATER, sep=';', decimal=',')
    data = data.values.tolist()

    x = [row[0:-1] for row in data]

    y = [[row[-1] for row in data]]

    lin_re = LinearRegression(x, y)

    ans = lin_re.linear_least_squares()
    print(ans)
    ans1 = lin_re.quadratic_least_squares()
    print(ans1)

    a = np.arange(194, 213, 0.1)
    plt.scatter(x, y)
    plt.plot(a, a * ans[1][0] + ans[0][0], 'g')
    plt.plot(a, a*a*ans1[2][0] + a*ans1[1][0] + ans1[0][0], 'r')
    plt.show()


    # Books attend grade dataset
    data = pandas.read_csv(FILE_DATA_BOOKS, sep=';')
    data = data.values.tolist()

    x = [row[0:-1] for row in data]

    y = [[row[-1] for row in data]]

    lin_re = LinearRegression(x, y)

    ans = lin_re.linear_least_squares()
    print(ans)

    # a = np.arange(194, 213, 0.1)
    ax = plt.axes(projection='3d')
    j = transpose(x)
    ax.scatter3D(j[0], j[1], y)
    # plt.plot(a, a * ans[1][0] + ans[0][0], 'g')
    plt.show()


