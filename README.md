# DualFlexKAN: Dual-Adaptive Kolmogorov-Arnold Networks

**DualFlexKAN** is a flexible and advanced PyTorch implementation of Kolmogorov-Arnold Networks (KAN). Unlike traditional implementations, DualFlexKAN offers **dual adaptability**, allowing independent control over both input transformation functions and output activation functions per layer.

This library extends the KAN paradigm to dense, convolutional (1D, 2D, 3D), and generative architectures (Autoencoders, Beta-VAEs).

## Key Features

*   **Dual Adaptability:** Configure independent strategies for input transformations (`per_input`, `per_channel`, `global`) and output activations (`per_neuron`, `fixed`).
*   **Rich Basis Functions:** Includes Polynomial, Legendre, Gegenbauer, Jacobi, B-Splines, and RBF (Radial Basis Functions).
*   **Convolutional Support:** Full support for **1D, 2D, and 3D KAN Convolutions**, enabling KAN-based CNNs for audio, vision, and volumetric data.
*   **Generative Models:** specialized wrappers for **Autoencoders** and **Beta-VAEs** with asymmetric encoder/decoder architectures.
*   **Advanced Initialization:** Robust initialization system (V2) with strategies like `he_normal`, `xavier`, and polynomial-specific scaling to ensure stable training.
*   **Flexible Regularization:** Fine-grained control over Dropout and Batch Normalization placement (pre or post-activation).

## File Structure

| File | Description |
| :--- | :--- |
| **`KAN_shared_poly7.py`** | **Core Library.** Contains `DualFlexKANLinear` and the main `DualFlexKAN` dense network implementation. |
| **`KAN_activation.py`** | Library of activation functions (Polynomial, Legendre, RBF, B-Spline, etc.). |
| **`KAN_initialization_v2.py`** | Advanced initialization logic offering 10 linear and 6 polynomial strategies. |
| **`DFKAN_conv.py* `** | **Convolutional KANs.** Implements `DualFlexKANConv1d/2d/3d` and full CNN wrappers. |
| **`DFKAN_autoencoder.py*`** | Wrapper for building flexible Autoencoders with different input/output sizes. |
| **`DFKAN_betaVAE.py*`** | Implementation of Beta-Variational Autoencoders using DualFlexKAN. |
| **`save_load_utils.py`** | Utilities for saving/loading models, configs, and experiments. |

* -> Not yet published
  
## Requirements

*   Python 3.8+
*   PyTorch
*   NumPy
*   Matplotlib
*   Scikit-learn
*   SciPy

## Usage Examples

### 1. Basic Dense Network (DualFlexKAN)
Create a standard KAN with polynomial inputs and RBF outputs.

```python
import torch
from KAN_shared_poly6 import DualFlexKAN

# Define architecture: Input(10) -> Hidden(32) -> Output(1)
model = DualFlexKAN(
    layer_sizes=[10, 32, 1],
    # Transform inputs using 3rd degree polynomials
    input_function_strategies=["per_input", "per_input"],
    input_activation_types=["polynomial", "polynomial"],
    input_activation_kwargs=[{'degree': 3}, {'degree': 3}],
    
    # Activate outputs using RBFs (Radial Basis Functions)
    output_function_strategies=["per_neuron", "fixed"],
    output_activation_types=["rbf", "linear"],
    output_activation_kwargs=[{'n_centers': 5}, {}]
)

x = torch.randn(16, 10)
y = model(x)
print(y.shape) # torch.Size([16, 1])
```

### 2. Convolutional Neural Network (CNN)
Build a 2D CNN using KAN layers for image processing.

```python
from DFKAN_conv import DualFlexKANCNN

# Create a CNN for MNIST-like data
model = DualFlexKANCNN(
    conv_configs=[
        {
            'in_channels': 1, 'out_channels': 16, 'kernel_size': 3,
            'input_activation_type': 'polynomial', # KAN convolution
            'pooling': {'type': 'max', 'kernel_size': 2}
        },
        {
            'in_channels': 16, 'out_channels': 32, 'kernel_size': 3,
            'input_activation_type': 'legendre',   # Legendre polynomials
            'pooling': {'type': 'max', 'kernel_size': 2}
        }
    ],
    dense_sizes=[64, 10] # Final classification layers
)
```

### 3. Autoencoder
Easily build an autoencoder for dimensionality reduction or super-resolution.

```python
from DFKAN_autoencoder import DualFlexKANAutoencoder

# Autoencoder: 784 (Input) -> 32 (Latent) -> 784 (Output)
ae = DualFlexKANAutoencoder(
    input_size=784,
    output_size=784,
    latent_size=32,
    encoder_hidden_sizes=[256, 128],
    decoder_hidden_sizes=[128, 256]
)

# Encode and Decode
x = torch.randn(8, 784)
reconstructed, latent = ae(x)
```

## Initialization Strategy

The library uses `KAN_initialization_v2.py` to handle the complexity of initializing polynomial and basis function parameters. You can initialize models smartly:

```python
from KAN_initialization_v2 import smart_kan_initialization

# Strategies: 'optimal', 'conservative', 'aggressive', 'stable', 'sparse'
smart_kan_initialization(model, strategy="stable")
```
## Installation
### From pip
pip install git+https://github.com/BioSiP/dfkan.git

### From sources
git clone https://github.com/BioSiP/dfkan.git
cd dfkan
pip install -e .


##  License
MIT License

Copyright (c) 2025 BioSiP Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

**Andres Ortiz and BioSiP group**
*July - November 2025*
