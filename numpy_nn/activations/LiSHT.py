import numpy as np
from ..layers import Layer

class LiSHT(Layer):
    '''
    '''
    def forward(self, x):
        self.tanh_x = np.tanh(x)
        self.last_x = x
        self.result = x * self.tanh_x
        return self.result

    def __call__(self, x):
        return self.forward(x)

    def backward(self, dL_dy):
        dy_partial = self.last_x * (1 - self.tanh_x**2)
        self.dy_dx = self.tanh_x +  dy_partial
        return self.dy_dx * dL_dy

class lisht(LiSHT):
    def __init__(self):
        pass
