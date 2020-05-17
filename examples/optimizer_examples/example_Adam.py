from numpy_nn import Neuron
from numpy_nn.losses import MSE
from numpy_nn.optimizers import Adam
from numpy_nn.models import Model

import numpy as np

# we will try to implment 2x + 1
elems = [x for x in range(0,100)]
x = np.array(elems).reshape((-1,1))
y = np.array([x + 1 for x in elems]).reshape((-1,1))

model = Model([
    Neuron(1,1)
])

mse = MSE()
adam = Adam(lr=0.09)

# setup loss and optimizer for the model
model.setup(loss_fn=mse, optimizer=adam)

# begin training the model
model.train(x,y, verbose=0,epochs=100,batch_size=10)

x = np.array([0,4,-1,-400,100]).reshape((-1,1))
out = model(x)
print("\nTESTING\n")
for i in range(x.shape[0]):
    print("Input:", x[i], "Output:", out[i])