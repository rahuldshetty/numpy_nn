from numpy_nn import Neuron
from numpy_nn.losses import MAE
from numpy_nn.optimizers import SGD
from numpy_nn.models import Model

import numpy as np

# we will try to implment 2x + 1
elems = [x for x in range(0,100)]
x = np.array(elems).reshape((-1,1))
y = np.array([2*x + 1 for x in elems]).reshape((-1,1))

model = Model([
    Neuron(1,2),
    Neuron(2,1)
])
mae = MAE()
sgd = SGD(lr=0.0002)

model.set_optimizer(sgd)

epochs = 10000

print("\nTRAINING\n")

for epoch in range(epochs):
    # batch 
    batches = x.shape[0]//10
    for batch in range(0,batches,10):
        x_sub = x[batch:batch+10]
        y_sub = y[batch:batch+10]
        # forward pass 
        out = model(x_sub)

        # find loss
        loss = mae(y_sub, out)

        # find gradient and optimize
        dL_dy = mae.gradient()
        model.backward(dL_dy)

        model.optimize()
    if epoch % 1000 == 0:
        print("Epoch {} Loss : {} ".format(epoch, loss))

x = np.array([0,4,-1,-400,100]).reshape((-1,1))
out = model(x)
print("\nTESTING\n")
for i in range(x.shape[0]):
    print("Input:", x[i], "Output:", out[i])