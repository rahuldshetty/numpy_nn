import numpy as np
from ..layers import Layer

class Sin(Layer):
    '''
    '''
    def forward(self, x):
        self.last_x = x
        self.result = np.sin(x)
        return self.result

    def __call__(self, x):
        return self.forward(x)

    def backward(self, dL_dy):
        self.dy_dx = np.cos(self.last_x)
        return self.dy_dx * dL_dy

class sin(Sin):
    def __init__(self):
        pass
