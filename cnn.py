from layers import *

class Cnn(Model):
    def __init__(self):
        layers = [
            Convolution("conv1",
                        kernel_height=5, 
                        kernel_width=5, 
                        channels_in=1, 
                        channels_out=32),
            ReLU(),
            MaxPooling(filter_size=2, stride=2),
            Convolution("conv2",
                        kernel_height=5, 
                        kernel_width=5, 
                        channels_in=32, 
                        channels_out=64),
            ReLU(),
            MaxPooling(filter_size=2, stride=2),
            Dense("dense1", channels_in=1024, channels_out=1024),
            ReLU(),
            Dense("dense2", channels_in=1024, channels_out=10),
            Softmax()
        ]
        super().__init__(layers)