from matrix import *
import copy

class LinearRegression:
    def __init__(self, x, y):
        """
        :param x: data input matrix
        """

        self.x = copy.deepcopy(x)
        self.x2 = copy.deepcopy(x)
        for i in range(len(x)):
            self.x[i].insert(0, 1)
            self.x2[i].append(self.x2[i][0]**2)
            self.x2[i].insert(0, 1)

        self.y = copy.deepcopy(y)
        self.y = transpose(self.y)

        self.coeficients = None
        self.linerar_coeficient = None
        self.angular_coeficient = None

    def linear_least_squares(self):
        self.coeficients = multiply(multiply(inverse(multiply(transpose(self.x), self.x)), transpose(self.x)), self.y)
        return self.coeficients

    def quadratic_least_squares(self):
        self.coeficients = multiply(multiply(inverse(multiply(transpose(self.x2), self.x2)), transpose(self.x2)), self.y)
        return self.coeficients

