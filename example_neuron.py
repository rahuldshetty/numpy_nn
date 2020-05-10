from numpy_nn import Neuron
from numpy_nn.losses import MAE
from numpy_nn.optimizers import SGD

import numpy as np

# we will try to implment 2x - 1
elems = [x for x in range(0,100)]
x = np.array(elems).reshape((-1,1))
y = np.array([x + 1 for x in elems]).reshape((-1,1))

neuron = Neuron(1, 1)
mae = MAE()
sgd = SGD(lr=0.0002)

neuron.set_optimizer(sgd)

epochs = 10000

print("\nTRAINING\n")

for epoch in range(epochs):
    # batch 
    batches = x.shape[0]//10
    for batch in range(0,batches,10):
        x_sub = x[batch:batch+10]
        y_sub = y[batch:batch+10]
        # forward pass 
        out = neuron(x_sub)

        # find loss
        loss = mae(y_sub, out)

        # find gradient and optimize
        dL_dy = mae.gradient()
        neuron.backward(dL_dy)

        neuron.optimize()
    if epoch % 1000 == 0:
        print("Epoch {} Loss : {} ".format(epoch, loss))

x = np.array([-1,4,1]).reshape((-1,1))
out = neuron(x)
print("\nTESTING\n")
for i in range(x.shape[0]):
    print("Input:", x[i], "Output:", out[i])