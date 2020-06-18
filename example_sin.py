from numpy_nn import Neuron
from numpy_nn.losses import MSE
from numpy_nn.optimizers import Adam
from numpy_nn.models import Model
from numpy_nn.activations import Sin

import numpy as np

# we will try to implment 2x + 1

def f(x):
    return np.sin(x)

elems = [x for x in range(0,100)]
x = np.array(elems).reshape((-1,1))
y = np.array([ f(x) for x in elems]).reshape((-1,1))


model = Model([
    Neuron(1,1,'N1'),
    Sin()
])

mse = MSE()
adam = Adam(lr=0.0001)

# setup loss and optimizer for the model
model.setup(loss_fn=mse, optimizer=adam)

# begin training the model
model.train(x,y, verbose=0,epochs=10000,batch_size=25)

x = np.array([0, 25, 50, 75, 100]).reshape((-1,1))
out = model(x)
print("\nTESTING\n")
for i in range(x.shape[0]):
    print("Input:", x[i], "Output:", out[i], "Expected sin:", np.sin(x[i]))