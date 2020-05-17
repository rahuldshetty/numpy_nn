from numpy_nn import Neuron
from numpy_nn.losses import MAE
from numpy_nn.optimizers import SGD
from numpy_nn.models import Model

import numpy as np

# we will try to implment 2x + 1
elems = [x for x in range(0,100)]
x = np.array(elems).reshape((-1,1))
y = np.array([x + 1 for x in elems]).reshape((-1,1))

model = Model([
    Neuron(1,1)
])
mae = MAE()
sgd = SGD(lr=0.00009)

# setup loss and optimizer for the model
model.setup(loss_fn=mae, optimizer=sgd)

# begin training the model
model.train(x,y, verbose=0,epochs=1000,batch_size=10)

model.save('model.pkl')
print('Saved model..')