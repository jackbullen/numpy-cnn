import numpy as np
import time

from dataset import Dataset
from cnn import *
from utils import OneHotEncoder

dataset = Dataset("mnist")
cnn = Cnn()
encoder = OneHotEncoder(10)

# Train set loop
BATCH_SIZE = 8

start = time.time()

loss = 0
for i, (image, label) in enumerate(zip(dataset.train_images, dataset.train_labels)):
    i += 1 

    # forward and backward pass
    target = encoder(label)
    pred = cnn.forward(image)
    loss += cnn.backward(target)

    # gradient updates every BATCH_SIZE iterations
    if i % BATCH_SIZE == 0: 
        cnn.update()
        num_correct = 0
        print(f"Mini-batch {i//8} loss: {loss}")
        loss = 0

    # evaluate every 100 iterations
    if i % 100 == 0:
        cnn.training = False
        for _ in range(30): 
            index = np.random.randint(len(dataset.test_images))
            image = dataset.test_images[index]
            label = dataset.test_labels[index]
            target = encoder(label)
            pred = cnn.forward(image)

            if np.argmax(pred) == label:
                num_correct += 1
        cnn.clear_gradients()
        print(f"num_correct: {num_correct}")
        cnn.training = True

    # save state every 1000 iterations
    if i % 1000 == 0: 
        print(f"Saving state after {i} iterations. Completed {i / len(dataset.train_images) * 100:.0f}% of test set")
        cnn.save_state()
        
end = time.time()
print("Finished training in", end-start)



# Test set loop
num_correct = 0
for image, label in zip(dataset.test_images, dataset.test_labels):
    target = encoder(label)

    pred = cnn.forward(image)
    if np.argmax(pred) == label:
        num_correct += 1

print("Accuracy =", num_correct / len(dataset.test_labels) * 100)
