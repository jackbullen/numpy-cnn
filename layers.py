import numpy as np

learning_rate = 0.001

class Convolution:
    def __init__(self, label, kernel_width, kernel_height, channels_in, channels_out, stride=1):
        self.label = label
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernels = 0.4 * np.random.rand(channels_out, channels_in, kernel_height, kernel_width) - 0.2
        self.stride = stride
        self.weight_gradients = []
        self.bias_gradients = []

    def __call__(self, x):
        channels_in, height, width = x.shape
        assert channels_in == self.channels_in, f"Input channel mismatch: {channels_in} != {self.channels_in}"
       
        x = np.tile(x, (self.channels_out, 1, 1, 1))

        output_height = ((height - self.kernel_height) // self.stride) + 1
        output_width = ((width - self.kernel_width) // self.stride) + 1
        out = np.zeros(shape=(self.channels_out, output_height, output_width))

        for row in range(0, output_height, self.stride):
            for col in range(0, output_width, self.stride):
                end_row = row + self.kernel_height
                end_col = col + self.kernel_width
                out[:, row, col] = np.sum(x[:,:,row:end_row,col:end_col] * self.kernels, axis=(1,2,3))

        self.previous_input = x
        self.previous_output = out
        return out
    
    def backward(self, output_gradient):
        #weight: convolution (X, K=output_gradient)
        input = self.previous_input#[0,:,:,:]

        _, _, height, width = input.shape
        _, k_height, k_width = output_gradient.shape

        output_gradient = np.expand_dims(output_gradient, 1)
        weight_gradient = np.zeros(shape=self.kernels.shape)

        for row in range(0, height-k_height+1, 1):#assume stride=1
            for col in range(0, width-k_width+1, 1):#assume stride=1
                end_row = row + k_height
                end_col = col + k_width
                weight_gradient[:,:,row,col] = np.sum(output_gradient * input[:,:,row:end_row,col:end_col], axis=(2,3))
        self.weight_gradients.append(weight_gradient)
        
        output_gradient = output_gradient.squeeze()
        
        #bias: output_gradient
        bias_gradient = output_gradient
        self.bias_gradients.append(bias_gradient)

        #input: padded convolution (output_gradient, K=reflected weights)  
        reflected_weights = self.kernels[:,:,::-1,::-1]

        _, depth, height, width = self.previous_input.shape
        input_gradient = np.zeros(shape=(depth, height, width))

        _, depth, k_height, k_width = reflected_weights.shape

        padded_output_gradient = np.zeros(shape=(output_gradient.shape + np.array([0, 2*(k_height-1), 2*(k_width-1)])))
        padded_output_gradient[:,k_height-1:-k_height+1,k_width-1:-k_width+1] = output_gradient
        padded_output_gradient = np.expand_dims(padded_output_gradient, 1)

        for row in range(0, height, 1):#assume stride=1
            for col in range(0, width, 1):#assume stride=1
                end_row = row + k_height
                end_col = col + k_width
                input_gradient[:,row,col] = np.sum(padded_output_gradient[:,:,row:end_row,col:end_col] * reflected_weights[:,:,:,:])

        return input_gradient

    def update(self):
        stacked_gradients = np.stack(self.weight_gradients)
        gradient = np.mean(stacked_gradients)
        self.kernels -= learning_rate * gradient
        self.clear_gradients()

    def save_state(self):
        np.save("weights/"+self.label+"_kernels", self.kernels)
    
    def clear_gradients(self):
        self.weight_gradients = []

class ReLU:
    def __call__(self, x):
        self.previous_input = x
        return np.maximum(0, x)
    
    def backward(self, output_gradient):
        local_gradient = np.array(self.previous_input > 0, dtype=np.int64)
        input_gradient = local_gradient * output_gradient
        return input_gradient

class MaxPooling:
    def __init__(self, filter_size, stride):
        self.filter_size = filter_size
        self.stride = stride

    def __call__(self, x):
        channels_in, width, height = x.shape

        output_height = ((height - self.filter_size) // self.stride) + 1
        output_width = ((width - self.filter_size) // self.stride) + 1

        out = np.zeros(shape=(channels_in, output_height, output_width))
        pooling_mask = np.zeros(shape=x.shape)

        for row in range(0, height, self.stride):
            for col in range(0, width, self.stride):
                end_row = row + self.filter_size
                end_col = col + self.filter_size
                max_indices_flattened = [np.argmax(x[dep,row:end_row,col:end_col]) 
                                         for dep in range(x.shape[0])]
                max_indices = [np.unravel_index(max_index, x[0,row:end_row,col:end_col].shape) 
                               for max_index in max_indices_flattened]  
                for dep in range(x.shape[0]):
                    out[dep,row//self.stride,col//self.stride] = x[dep,row:end_row,col:end_col][*max_indices[dep]]
                    pooling_mask[dep,row:end_row,col:end_col][*max_indices[dep]] = 1

        self.pooling_mask = pooling_mask
        self.previous_input = x
        self.previous_output = out
        return out

    def backward(self, output_gradient):
        output_gradient = output_gradient.reshape(self.previous_output.shape)
        repeated_output_gradient = np.repeat(output_gradient, 2, axis=1)
        repeated_output_gradient = np.repeat(repeated_output_gradient, 2, axis=2)
        input_gradient = repeated_output_gradient * self.pooling_mask
        return input_gradient

class Dense:
    def __init__(self, label, channels_in, channels_out):
        self.label = label
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
    
    def backward(self, output_gradient):
        weight_gradient = np.outer(output_gradient, self.previous_input)
        self.weight_gradients.append(weight_gradient)
        bias_gradient = output_gradient
        self.bias_gradients.append(bias_gradient)

        input_gradient = output_gradient @ self.weights
        return input_gradient

    def update(self):
        weight_gradient = np.mean(np.array(self.weight_gradients), axis=0)
        self.weights -= learning_rate * weight_gradient
        bias_gradient = np.mean(np.array(self.bias_gradients), axis=0)
        self.biases -= learning_rate * bias_gradient
        self.clear_gradients()

    def save_state(self):
        np.save("weights/"+self.label+"_weights", self.weights)
        np.save("weights/"+self.label+"_biases", self.biases)

    def clear_gradients(self):
        self.weight_gradients = []
        self.bias_gradients = []

class Dropout:
    def __init__(self, probability):
        self.probability = probability
        
    def __call__(self, x):
        mask = np.random.binomial(n=1, p=1-self.probability, size=x.shape)
        output = x * mask
        self.previous_mask = mask
        return output

    def backward(self, output_gradient):
        input_gradient = output_gradient * self.previous_mask
        return input_gradient

class Softmax:
    def __call__(self, x):
        exp = np.exp(x)
        activation = exp / np.sum(exp)
        return activation
    
class CrossEntropyLoss:
    def __call__(self, x, target):
        return -np.dot(target, np.log(x))
    
class Model:
    def __init__(self, layers):
        self.layers = layers
        self.loss_function = CrossEntropyLoss()
        self.training = True
        self.losses = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        self.previous_output = x
        return x
    
    def backward(self, target):
        gradient = self.previous_output - target#assume softmax + cross entropy
        for layer in self.layers[-2::-1]:
            gradient = layer.backward(gradient)
        loss = self.loss_function(self.previous_output, target)
        self.losses.append(loss)
        return loss

    def update(self):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update()
    
    def save_state(self):
        for layer in self.layers:
            if hasattr(layer, "save_state"):
                layer.save_state()

    def clear_gradients(self):
        for layer in self.layers:
            if hasattr(layer, "clear_gradients"):
                layer.clear_gradients()