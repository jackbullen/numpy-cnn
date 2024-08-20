import numpy as np

from cnn import *
from layers import Model
from utils import OneHotEncoder

class TestCnn(Model):
    def __init__(self):
        layers = [
            Convolution("conv",
                        kernel_height=3, 
                        kernel_width=3, 
                        channels_in=1, 
                        channels_out=4),
            ReLU(),
            MaxPooling(filter_size=2, stride=2),
            Dense("dense", channels_in=4, channels_out=2),
            ReLU(),
            Softmax()
        ]
        super().__init__(layers)
    

encoder = OneHotEncoder(2)

cnn = TestCnn()

# set conv weights
cnn.layers[0].kernels = np.tile(np.array([[1, 0, 0], [0, 0, -1], [1, 1, 0]]), (4, 1, 1, 1))

# set dense weights
cnn.layers[3].weights = np.array([[1, 1, -1, -1], [1, 1, 0, 1]])
cnn.layers[3].biases = np.array([1, 0])

x = np.array([[1, 2, 3, 4], 
              [5, 6, 7, 8], 
              [9, 10, 11, 12], 
              [13, 14, 15, 16]], 
             dtype=np.float64).reshape(1, 4, 4)

x_out = cnn.forward(x)

label = 0
target = encoder(label)

loss = cnn.backward(target)
print("Loss =", loss)


### Forward

# 1. Input 
# [[ 1,  2,  3,  4], 
#  [ 5,  6,  7,  8], 
#  [ 9, 10, 11, 12], 
#  [13, 14, 15, 16]]

# 2. Convolution(cin=1, cout=4, K=3x3)
# [[[13. 15.]
#   [21. 23.]]
#  [[13. 15.]
#   [21. 23.]]
#  [[13. 15.]
#   [21. 23.]]
#  [[13. 15.]
#   [21. 23.]]]

# 3. Relu (unchanged)

# 4. MaxPooling(2x2)
# [[[23.]]
#  [[23.]]
#  [[23.]]
#  [[23.]]]

# 5. Dense
# [ 1. 69.]

# 6. Relu (unchanged)

# 7. Softmax
# [2.93748211e-30 1.00000000e+00]


### Backward

# 1.
