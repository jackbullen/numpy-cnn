# numpy-cnn

Convolution network implemented in NumPy

### Things to do
- Weight adjustments (momentum, variable learning rate, ...)
- Regularization (l1, l2, dropout, ...)
- Evaluation (precision, recall, confusion matrix, ...)
- Visualization

### Layout
- Layer classes and Model are defined in [layers.py](./layers.py)
- Cnn class is defined in [cnn.py](./cnn.py)
- Training loop for Cnn class is in [cnn.ipynb](./cnn.ipynb) and [main.py](./main.py)
- Tests to check for bugs are in [test_cnn](./test_cnn.py) and [test_dense](./test_dense.py)