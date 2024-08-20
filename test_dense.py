"""
Interestingly this model achieved a higher accuracy and trains far far quicker than Cnn
(something in convultion layer is implemented slowly or convolutions just take a long time). 
Accuracy could be due to training the cnn model being more difficult (bigger model, more weights to learn) 
and with only basic gradient descent it is not converging well.

This file has not been updated to use backward method in layers or Model class.
"""

import numpy as np
from dataset import Dataset
import time 

learning_rate = 0.01

class Softmax:
    def __call__(self, x):
        exp = np.exp(x)
        activation = exp / np.sum(exp)
        return activation

class CrossEntropyLoss:
    def __call__(self, x, target):
        return -np.dot(target, np.log(x))

class OneHotEncoder:
    def __init__(self, n):
        self.n = n

    def __call__(self, label):
        x = np.zeros(self.n)
        x[label] = 1.0
        return x

class ReLU:
    def __call__(self, x):
        self.previous_input = x
        return np.maximum(0, x)
    
class Dense:
    def __init__(self, channels_in, channels_out):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.weights = 0.4 * np.random.rand(channels_out, channels_in) - 0.2
        self.biases = np.array(channels_out*[0.1])
        self.weight_gradients = []
        self.bias_gradients = []

    def __call__(self, x):
        x = x.flatten()
        out = self.weights @ x + self.biases
        self.previous_input = x
        self.previous_output = out
        return out
    
    def update(self):
        weight_gradients = np.mean(np.array(self.weight_gradients), axis=0)
        self.weights -= learning_rate * weight_gradients
        bias_gradients = np.mean(np.array(self.bias_gradients), axis=0)
        self.biases -= learning_rate * bias_gradients

        self.clear_gradients()

    def clear_gradients(self):
        self.weight_gradients = []
        self.bias_gradients = []

class Nn:
    def __init__(self):
        self.dense1 = Dense(channels_in=784, channels_out=256)
        self.relu1 = ReLU()
        self.dense2 = Dense(channels_in=256, channels_out=128)
        self.relu2 = ReLU()
        self.dense3 = Dense(channels_in=128, channels_out=10)
        self.softmax = Softmax()
        self.loss_function = CrossEntropyLoss()
        self.training = True

    def forward(self, x):
        # print("Forward pass:")
        # print("Input", x.shape)
        x = self.dense1(x)
        # print("Dense1", x.shape)
        x = self.relu1(x)
        # print("Relu1", x.shape)
        x = self.dense2(x)
        # print("Dense2", x.shape)
        x = self.relu2(x)
        # print("Relu2", x.shape)
        x = self.dense3(x)
        # print("Dense3", x.shape)
        x = self.softmax(x)
        # print("Softmax", x.shape)
        self.previous_output = x
        return x
    
    def backward(self, target):
        # Network output gradient
        loss = self.loss_function(self.previous_output, target)
        # print(f"\nBackward pass with loss = {loss:.2f}:")

        dLdO_network = self.previous_output - target
        # print("dLdO_network", dLdO_network.shape)

        # Dense 3
        dense3_weight_gradient = np.outer(dLdO_network, self.dense3.previous_input)
        dense3_bias_gradient = dLdO_network
        if self.training:
            self.dense3.weight_gradients.append(dense3_weight_gradient)
            self.dense3.bias_gradients.append(dense3_bias_gradient)

        dLdX_dense3 = dLdO_network @ self.dense3.weights
        # print("dLdX_dense3", dLdX_dense3.shape)

        # Relu 2
        dLdO_relu2 = np.array(self.relu2.previous_input > 0, dtype=float)
        dLdX_relu2 = dLdX_dense3 * dLdO_relu2
        # print("dLdX_relu2", dLdX_relu2.shape)

        # Dense 2
        dense2_weight_gradient = np.outer(dLdX_relu2, self.dense2.previous_input)
        dense2_bias_gradient = dLdX_relu2
        if self.training:
            self.dense2.weight_gradients.append(dense2_weight_gradient)
            self.dense2.bias_gradients.append(dense2_bias_gradient)

        dLdX_dense2 = dLdX_relu2 @ self.dense2.weights
        # print("dLdX_dense2", dLdX_dense2.shape)

        # Relu 1
        dLdO_relu1 = np.array(self.relu1.previous_input > 0, dtype=float)
        dLdX_relu1 = dLdX_dense2 * dLdO_relu1
        # print("dLdX_relu1", dLdX_relu1.shape)

        # Dense 1
        dense1_weight_gradient = np.outer(dLdX_relu1, self.dense1.previous_input)
        dense1_bias_gradient = dLdX_relu1
        if self.training:
            self.dense1.weight_gradients.append(dense1_weight_gradient)
            self.dense1.bias_gradients.append(dense1_bias_gradient)

        dLdX_dense1 = dLdX_relu1 @ self.dense1.weights
        # print("dLdX_dense1", dLdX_dense1.shape)

        return loss

    def update(self):
        self.dense1.update()
        self.dense2.update()
        self.dense3.update()
    
    def clear_gradients(self):
        self.dense1.clear_gradients()
        self.dense2.clear_gradients()
        self.dense3.clear_gradients()

nn = Nn()

dataset = Dataset("mnist")
encoder = OneHotEncoder(10)

# for i in range(10):
#     x = dataset.train_images[i]
#     label = dataset.train_labels[i]

#     target = encoder(label)

#     pred = nn.forward(x)
#     loss = nn.backward(target)
#     nn.update()

##
# Training
start = time.time()

loss = 0
for i, (image, label) in enumerate(zip(dataset.train_images, dataset.train_labels)):
    
    # forward and backward pass
    target = encoder(label)
    pred = nn.forward(image)
    loss += nn.backward(target)
    # print("Train:", np.argmax(pred), label)

    # update after a batch
    if (i+1) % 1 == 0: 
        nn.update()
        num_correct = 0
        
        print(f"Mini-batch {i//10} loss: {loss}")
        
        loss = 0

    if (i+1) % 100 == 0:
        nn.training = False
        for _ in range(10):
            index = np.random.randint(len(dataset.test_images))
            image = dataset.test_images[index]
            label = dataset.test_labels[index]
            target = encoder(label)
            pred = nn.forward(image)

            if np.argmax(pred) == label:
                num_correct += 1
        nn.clear_gradients()
        print(f"num_correct: {num_correct}")
        nn.training = True

    # progress report every 1000 iterations
    # if i % 1000 == 0: 
        # print(i / len(dataset.train_images) * 100)
        
end = time.time()
print("Finished training in", end-start)

# Evaluation
num_correct = 0
for image, label in zip(dataset.test_images, dataset.test_labels):
    target = encoder(label)

    pred = nn.forward(image)
    if np.argmax(pred) == label:
        num_correct += 1

print(num_correct, len(dataset.test_labels))
