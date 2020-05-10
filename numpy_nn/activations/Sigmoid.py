import numpy as np

class Sigmoid():
    
    def forward(self, x):
        self.last_x = x
        self.result = 1.0 / (1.0 + np.exp(-x))
        return self.result

    def __call__(self, x):
        return self.forward(x)

    def backward(self, dL_dy):
        self.dy_dx = self.result * (1 - self.result)
        return self.dy_dx * dL_dy
