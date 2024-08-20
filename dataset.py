import numpy as np

IMAGE_WIDTH, IMAGE_HEIGHT = 28, 28
IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT

class Dataset:
    def __init__(self, datadir, batch_size=10):
        self.train_images = load_images(f"{datadir}/train_images")
        self.train_labels = load_labels(f"{datadir}/train_labels")
        self.test_images = load_images(f"{datadir}/test_images")
        self.test_labels = load_labels(f"{datadir}/test_labels")
        self.batch_size = batch_size
        self.train_size = len(self.train_images)

    def __iter__(self):
        return self
    
    def __next__(self):
        return self._random_training_batch()
    
    def _random_training_batch(self):
        images = []
        labels = []
        for _ in range(self.batch_size):
            random_index = np.random.randint(0, self.train_size)
            images.append(self.train_images[random_index])
            labels.append(self.train_labels[random_index])
        return images, labels

def load_labels(filepath):
    with open(filepath, 'rb') as f:
        label_bytes = f.read()
    return [int(byte) for byte in label_bytes]

def load_images(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    return [np.array([int(x) / 255 for x in image_bytes[i*IMAGE_BYTES:(i+1)*IMAGE_BYTES]]
                     ).reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH) 
                     for i in range(0, len(image_bytes)//IMAGE_BYTES)]