import numpy as np
import matplotlib.pyplot as plt
from dataset import Dataset

dataset = Dataset("mnist")
image = dataset.test_images[0]

# C[i, j] = sum_l sum_m I[i+l, j+m] * K[l, m]
def conv(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape
    
    feature_map = np.array([np.zeros(width - k_width + 1) 
                            for _ in range(height - k_height + 1)])
    for i in range(width - k_width + 1):
        for j in range(height - k_height + 1):
            feature_map[j, i] = 0
            for l in range(k_width):
                for m in range(k_height):
                    feature_map[j, i] += image[j+m, i+l] * kernel[m, l]
    
    return feature_map

vertical_map = conv(image,   np.array([[ 1,  0, -1], 
                                       [ 1,  0, -1],
                                       [ 1,  0, -1]]))
horizontal_map = conv(image, np.array([[ 1,  1,  1], 
                                       [ 0,  0,  0],
                                       [-1, -1, -1]]))

plt.figure(figsize=(20, 8))
plt.subplot(131); plt.imshow(image); plt.title("Original")
plt.subplot(132); plt.imshow(horizontal_map); plt.title("Horizontal Kernel")
plt.subplot(133); plt.imshow(vertical_map); plt.title("Vertical Kernel")
print(image.shape, vertical_map.shape)
plt.show()

