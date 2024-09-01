import torch
import numpy as np
import time
from layers import Convolution

np.random.seed(100)

def test_correctness(kernel_size=2, channels_in=3, channels_out=32, image_size=100, padding=0, stride=1):
    x = np.random.randn(channels_in, image_size, image_size)
    torch_x = torch.tensor(x, requires_grad=True)
    # 1. Convolution class
    # 2. PyTorch Conv2d class
    # define layer
    #1.
    conv = Convolution("conv", kernel_size, kernel_size, channels_in, channels_out, stride)
    #2.
    torch_conv = torch.nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False)
    torch_conv.weight.data = torch.tensor(conv.kernels)
    # forward pass
    #1.
    out = conv(x)
    #2.
    torch_out = torch_conv(torch_x)
    # backward pass
    #1.
    conv.backward(np.ones(out.shape))
    #2. 
    # sum them because torch requires scalar to backward
    # this is consistent with np.ones out gradient above
    torch.sum(torch_out).backward()

    assert (out - torch_out.detach().numpy() < 1e-8).all(), f"Forward pass disagreement: Convolution(K={kernel_size}x{kernel_size}, cin={channels_in}, cout={channels_out}, s={stride})"
    assert (conv.weight_gradients - torch_conv.weight.grad.detach().numpy() < 1e-8).all(), f"Backward pass disagreement: Convolution(K={kernel_size}x{kernel_size}, cin={channels_in}, cout={channels_out}, s={stride})"

def test_speed(kernel_size=2, channels_in=3, channels_out=32, image_size=100, padding=0, stride=1):
    conv = Convolution("conv", kernel_size, kernel_size, channels_in, channels_out, stride)
    torch_conv = torch.nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False)
    torch_conv.weight.data = torch.tensor(conv.kernels)

    x = np.random.randn(channels_in, image_size, image_size)

    out = conv(x)

    def forward_pass():
        start = time.time()
        conv(x)
        end = time.time()
        return end - start
    times = [forward_pass() for _ in range(10)]
    print(f"Forward: mu = {np.mean(times):.2f}, sigma = {np.std(times):.4f}")

    def backward_pass():
        start = time.time()
        conv.backward(np.ones(out.shape))
        end = time.time()
        return end - start 
    times = [backward_pass() for _ in range(10)]
    print(f"Backward: mu = {np.mean(times):.2f}, sigma = {np.std(times):.4f}")

if __name__ == "__main__":
    test_correctness()
    # test_correctness(kernel_size=5, channels_in=32, channels_out=64, image_size=100, padding=0, stride=1)
    test_speed(kernel_size=5, channels_in=32, channels_out=64, image_size=10, padding=0, stride=1)
