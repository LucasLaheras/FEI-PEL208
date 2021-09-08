import numpy as np
from loss import *


class Activation:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.gradient_inputs = None


class Activation_softmax(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalized_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = normalized_values

    def backward(self, gradient):
        self.gradient_inputs = np.empty_like(gradient)

        for index, (single_output, single_derivate) in enumerate(zip(self.output, gradient)):

            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_derivate.T)

            self.gradient_inputs[index] = np.dot(jacobian_matrix, single_derivate)

    def prediction(self, outputs):
        return np.argmax(outputs, axis=1)


class Activation_ReLU(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, gradient):
        self.gradient_inputs = gradient.copy()

        self.gradient_inputs[self.inputs < 0] = 0

    def prediction(self, outputs):
        return outputs


class Activation_Sigmoid(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, gradient):
        self.gradient_inputs = gradient * (1 - self.output) * self.output

    def prediction(self, outputs):
        return (outputs > 0.5) * 1


class Activation_Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, gradient):
        self.gradient_inputs = gradient.copy()

    def prediction(self, outputs):
        return outputs


class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.output = None
        self.gradient_inputs = None
        self.activation = Activation_softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, gradient, y_true):
        samples = len(gradient)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.gradient_inputs = gradient.copy()
        self.gradient_inputs[range(samples), y_true] -= 1
        self.gradient_inputs = self.gradient_inputs / samples


class q_gabor():
    def __init__(self, q, alpha=0.3, f=0.08, theta=0, k=1):
        """"
            :param alpha:
            :param q: opening
            :param f:
            :param theta: angle
            :param k:
        """
        self.q = q
        self.alpha = alpha
        self.f = f
        self.theta = theta
        self.k = k

    @staticmethod
    def sinusoidal_function(f, x):  # s(X) = e^(2*π*f*X*i)
        s = pow(np.e, 2 * np.pi * f * x * 1j)
        return s

    @staticmethod
    def q_exponential_function(x, q):  # w(X) =1/(1+(1-q)*X^2)^(1/(1-q))
        if q == 1:
            w = pow(np.e, -np.pi * pow(x, 2))
        else:
            w = 1 / pow(1 + (1 - q) * x * x, 1 / (1 - q))
        return w

    def q_gabor_1d(self, x, alpha, q, f, theta, k):  # g(X)=k*e^(theta*i)*w(alpha*X)*s(X)
        sinusoidal = self.sinusoidal_function(f, alpha * x)
        q_exponencial = self.q_exponential_function(x, q)
        g = k * pow(np.e, (theta * 1j)) * sinusoidal * q_exponencial
        return g

    @staticmethod
    def q_gabor_2d(x, y, q, k, u, v, p, a, b):
        """"
            :param x = data
            :param y = data
            :param q = opening
            :param k = amplitude
            :param u = X filter frequency
            :param v = Y filter frequency
            :param p = filter phase
            :param a = envelope
            :param b = envelope
        """
        xo = yo = 0
        w = k * (1 / ((1 + (1 - q) * ((a ** 2 * (x - xo) ** 2 + b ** 2 * (y - yo) ** 2))) ** (1 / (1 - q)))) #<- formula diferente!!!!
        s = np.exp((2 * np.pi * (u * x + v * y) + p) * 1j)
        g = w * s
        return g

    def q_gabor_activation(self, x):
        """"
            :param x: data
        """
        x = tf.cast(x, dtype=tf.complex128)
        g = self.q_gabor_1d(x, self.alpha, self.q, self.f, self.theta, self.k)
        return tf.cast(g, dtype=tf.float32)