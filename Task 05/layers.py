import numpy as np


class Layer_Input:
    def __init__(self, name='input_layer'):
        self.name = name

    def forward(self, inputs, training):
        self.output = inputs


class Layer_Dense:
    def __init__(self, number_inputs, number_neurons, name='my_layer', weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights using random numbers using gaussian distribution and normalized in range [0 0.10]
        self.weights = 0.10 * np.random.randn(number_inputs, number_neurons)
        self.biases = np.zeros((1, number_neurons))

        # Regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        self.inputs = None
        self.output = None
        self.gradient_weights = None
        self.gradient_biases = None
        self.gradient_inputs = None

        self.name = name

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, gradient):
        self.gradient_weights = np.dot(self.inputs.T, gradient)
        self.gradient_biases = np.sum(gradient, axis=0, keepdims=True)

        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.gradient_weights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.gradient_weights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.gradient_biases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.gradient_biases += 2 * self.bias_regularizer_l2 * self.biases

        self.gradient_inputs = np.dot(gradient, self.weights.T)


class Layer_Dropout:
    def __init__(self, rate, name='my_dropout'):
        self.rate = 1 - rate
        self.inputs = None
        self.gradient_inputs = None
        self.binary_mask = None
        self.output = None
        self.name = name

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, gradient):
        self.gradient_inputs = gradient * self.binary_mask


class Pooling2D:
    def __init__(self, kernel_size=(2, 2), method=np.nanmax, pad=True):
        self.kernel_size = kernel_size
        self.pad = pad
        # numpy function np.nanmean or np.nanmax
        self.method = method

    def forward(self, single_input):
        m, n = single_input.shape[:2]
        ky, kx = self.kernel_size

        _ceil = lambda x, y: int(np.ceil(x / float(y)))

        if self.pad:
            ny = _ceil(m, ky)
            nx = _ceil(n, kx)
            size = (ny * ky, nx * kx) + single_input.shape[2:]
            mat_pad = np.full(size, np.nan)
            mat_pad[:m, :n, ...] = single_input
        else:
            ny = m // ky
            nx = n // kx
            mat_pad = single_input[:ny * ky, :nx * kx, ...]

        new_shape = (ny, ky, nx, kx) + single_input.shape[2:]

        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

        return result


def convolution(single_input, kernel, stride, padding):
    single_input = np.pad(single_input, [(padding, padding), (padding, padding)], mode='constant', constant_values=0)

    kernel_height, kernel_width = kernel.shape
    padded_height, padded_width = single_input.shape

    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width)).astype(np.float32)

    for y in range(0, output_height):
        for x in range(0, output_width):
            output[y][x] = np.sum(single_input[y * stride:y * stride + kernel_height,
                                     x * stride:x * stride + kernel_width] * kernel).astype(np.float32)
    return output


class Layer_Convolution2D:
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', name='my_convolution'):
        self.gradient_inputs = None
        self.inputs = None
        self.output = None
        self.name = name
        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size
        self.filters = filters
        self.kernels = list()

        self.weights = 0.10 * np.random.randn(number_inputs, number_neurons)
        self.biases = np.zeros((1, number_neurons))
        self.input = None
        self.output = None
        self.name = name

    def forward(self, inputs, training):
        self.output = list()

        for i in inputs:
            self.output.append(list())
            for k in self.kernels:
                self.output[i].append(convolution(i, k, self.strides, np.max(k.shape) // 2))

    def backward(self, dout, cache):

        dx, dw, db = None, None, None

        # Récupération des variables
        x, w, b, conv_param = cache
        pad = conv_param['pad']
        stride = conv_param['stride']

        # Initialisations
        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.zeros_like(b)

        # Dimensions
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, H_, W_ = dout.shape

        # db - dout (N, F, H', W')
        # On somme sur tous les éléments sauf les indices des filtres
        db = np.sum(dout, axis=(0, 2, 3))

        # dw = xp * dy
        # 0-padding juste sur les deux dernières dimensions de x
        xp = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')

        # Version sans vectorisation
        for n in range(N):  # On parcourt toutes les images
            for f in range(F):  # On parcourt tous les filtres
                for i in range(HH):  # indices du résultat
                    for j in range(WW):
                        for k in range(H_):  # indices du filtre
                            for l in range(W_):
                                for c in range(C):  # profondeur
                                    dw[f, c, i, j] += xp[n, c, stride * i + k, stride * j + l] * dout[n, f, k, l]

        # dx = dy_0 * w'
        # Valide seulement pour un stride = 1
        # 0-padding juste sur les deux dernières dimensions de dy = dout (N, F, H', W')
        doutp = np.pad(dout, ((0,), (0,), (WW - 1,), (HH - 1,)), 'constant')

        # 0-padding juste sur les deux dernières dimensions de dx
        dxp = np.pad(dx, ((0,), (0,), (pad,), (pad,)), 'constant')

        # filtre inversé dimension (F, C, HH, WW)
        w_ = np.zeros_like(w)
        for i in range(HH):
            for j in range(WW):
                w_[:, :, i, j] = w[:, :, HH - i - 1, WW - j - 1]

        # Version sans vectorisation
        for n in range(N):  # On parcourt toutes les images
            for f in range(F):  # On parcourt tous les filtres
                for i in range(H + 2 * pad):  # indices de l'entrée participant au résultat
                    for j in range(W + 2 * pad):
                        for k in range(HH):  # indices du filtre
                            for l in range(WW):
                                for c in range(C):  # profondeur
                                    dxp[n, c, i, j] += doutp[n, f, i + k, j + l] * w_[f, c, k, l]
        # Remove padding for dx
        dx = dxp[:, :, pad:-pad, pad:-pad]

        return dx, dw, db


