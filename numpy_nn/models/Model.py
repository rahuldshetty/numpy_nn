import numpy as np

class Model():
    
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer in self.layers:
            y = layer(x)
            x = y
        return y
    
    def backward(self, dL):
        for layer in reversed(self.layers):
            dL = layer.backward(dL)
        return dL
    
    def optimize(self):
        for layer in self.layers:
            layer.optimize()

    def set_optimizer(self, optimizer):
        for layer in self.layers:
            layer.set_optimizer(optimizer)