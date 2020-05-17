from numpy_nn import Neuron
from numpy_nn.losses import MSE
from numpy_nn.optimizers import Adam
from numpy_nn.models import Model
from numpy_nn.activations import Sigmoid, Swish

import numpy as np

def f(x):
    return 1 if x >= 50 else 0

elems = [x for x in range(0,100)]
x = np.array(elems).reshape((-1,1))
y = np.array([ f(x) for x in elems]).reshape((-1,1))


model = Model([
    Neuron(1,1),
    Swish(),
    Neuron(1,1),
    Sigmoid()
])

mse = MSE()
adam = Adam(lr=0.0009)

# setup loss and optimizer for the model
model.setup(loss_fn=mse, optimizer=adam)

# begin training the model
model.train(x,y, verbose=0,epochs=1000,batch_size=25)

x = np.array([0, 25, 50, 75, 100]).reshape((-1,1))
out = model(x)
print("\nTESTING\n")
for i in range(x.shape[0]):
    print("Input:", x[i], "Output:", out[i])