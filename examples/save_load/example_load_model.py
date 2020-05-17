from numpy_nn.models import Model
import numpy as np

model = Model.load('model.pkl')
print('Model Loaded...')

x = np.array([0,4,-1,-400,100]).reshape((-1,1))
out = model(x)
print("\nTESTING\n")
for i in range(x.shape[0]):
    print("Input:", x[i], "Output:", out[i])