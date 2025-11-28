# Deep Neural Network Application

## Scalable L-Layer Neural Network from Scratch

## Overview

A fully-connected deep neural network implementation built from scratch using Python and NumPy. This project provides a modular, scalable architecture supporting arbitrary network depth, demonstrating how deep networks achieve superior performance over shallow architectures through hierarchical feature learning.

## Author

- **Anmol Dubey**

## Features

| Feature | Description |
|---------|-------------|
| Configurable Depth | Support for arbitrary L-layer architectures |
| Modular Design | Reusable forward/backward propagation building blocks |
| Multiple Activations | ReLU for hidden layers, sigmoid for output |
| Vectorized Operations | Efficient NumPy-based matrix computations |
| Cache Management | Stored intermediate values for efficient backpropagation |
| Cost Tracking | Training progress visualization |

## Architecture

The network implements the pattern `[LINEAR→RELU]*(L-1) → LINEAR→SIGMOID`:

```
Input Layer    Hidden Layers (ReLU)         Output Layer (Sigmoid)
    
   X          ┌─────┐   ┌─────┐   ┌─────┐      ┌─────┐
 (12288)  ──► │  20 │──►│  7  │──►│  5  │ ──►  │  1  │ ──► ŷ
              └─────┘   └─────┘   └─────┘      └─────┘
              Layer 1   Layer 2   Layer 3      Layer 4

              W1, b1    W2, b2    W3, b3       W4, b4
```

Default configuration: `[12288, 20, 7, 5, 1]` (4-layer network)

## Mathematical Foundations

### Forward Propagation

For each layer l = 1, 2, ..., L:

```
Linear:     Z[l] = W[l] · A[l-1] + b[l]
Activation: A[l] = g[l](Z[l])
```

Where:
- `g[l] = ReLU` for l = 1, ..., L-1 (hidden layers)
- `g[L] = sigmoid` for output layer

### Cost Function

Binary cross-entropy loss:

```
J = -(1/m) · Σᵢ[y⁽ⁱ⁾log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾)log(1-ŷ⁽ⁱ⁾)]
```

### Backward Propagation

Output layer gradient initialization:
```
dA[L] = -(y/A[L]) + (1-y)/(1-A[L])
```

For each layer l = L, L-1, ..., 1:
```
dZ[l] = dA[l] * g'[l](Z[l])
dW[l] = (1/m) · dZ[l] · A[l-1].T
db[l] = (1/m) · Σ(dZ[l])
dA[l-1] = W[l].T · dZ[l]
```

### Parameter Update

Gradient descent:
```
W[l] = W[l] - α · dW[l]
b[l] = b[l] - α · db[l]
```

## Implementation Details

### Core Functions

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `initialize_parameters_deep` | Initialize weights/biases for L layers | `layer_dims` list | Parameters dict |
| `L_model_forward` | Complete forward pass | X, parameters | AL, caches |
| `compute_cost` | Cross-entropy loss | AL, Y | Cost scalar |
| `L_model_backward` | Complete backward pass | AL, Y, caches | Gradients dict |
| `update_parameters` | Gradient descent step | parameters, grads, α | Updated parameters |

### Building Block Functions

| Function | Purpose |
|----------|---------|
| `linear_forward` | Z = W·A + b computation |
| `linear_activation_forward` | Linear + activation (single layer) |
| `linear_backward` | Gradient computation for linear part |
| `linear_activation_backward` | Gradient computation with activation |

### Model Functions

| Function | Architecture | Use Case |
|----------|--------------|----------|
| `two_layer_model` | LINEAR→RELU→LINEAR→SIGMOID | Baseline comparison |
| `L_layer_model` | [LINEAR→RELU]*(L-1)→LINEAR→SIGMOID | Deep network |

## Dataset

**Cat vs Non-Cat Image Classification**

| Property | Training Set | Test Set |
|----------|--------------|----------|
| Examples | 209 | 50 |
| Image Size | 64×64×3 | 64×64×3 |
| Flattened Features | 12,288 | 12,288 |
| Classes | 2 (cat/non-cat) | 2 |

### Preprocessing

```python
# Flatten images: (m, 64, 64, 3) → (12288, m)
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T

# Normalize pixel values to [0, 1]
train_x = train_x_flatten / 255.
```

## Usage

### Running the Model

```bash
python deepNeuralNetworkApplication.py
```

### Training a Custom Architecture

```python
# Define architecture
layers_dims = [12288, 20, 7, 5, 1]  # 4-layer network

# Train model
parameters, costs = L_layer_model(
    train_x, 
    train_y, 
    layers_dims,
    learning_rate=0.0075,
    num_iterations=2500,
    print_cost=True
)

# Evaluate
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
```

### Comparing Architectures

```python
# Two-layer model (baseline)
params_2layer, costs_2layer = two_layer_model(
    train_x, train_y, 
    layers_dims=(12288, 7, 1),
    num_iterations=2500
)

# Four-layer model (deep)
params_4layer, costs_4layer = L_layer_model(
    train_x, train_y,
    layers_dims=[12288, 20, 7, 5, 1],
    num_iterations=2500
)
```

## Results

### Training Progress (4-Layer Model)

```
Cost after iteration 0: 0.771749
Cost after iteration 100: 0.672053
Cost after iteration 200: 0.648263
...
Cost after iteration 2400: 0.048554
Cost after iteration 2499: 0.048089
```

### Performance Comparison

| Model | Layers | Training Accuracy | Test Accuracy |
|-------|--------|-------------------|---------------|
| Two-Layer | 2 | ~100% | ~72% |
| Four-Layer | 4 | ~98% | ~80% |

The deeper network achieves better generalization despite similar training performance.

## File Structure

```
deepNeuralNetworkApplication/
├── README.md                           # This file
├── deepNeuralNetworkApplication.py     # Main implementation
├── projectDescription                  # Project summary
├── dnn_utils.py                        # Activation functions
│   ├── sigmoid()
│   ├── sigmoid_backward()
│   ├── relu()
│   └── relu_backward()
├── testCases.py                        # Unit test cases
├── public_tests.py                     # Validation tests
└── datasets/
    └── train_catvnoncat.h5             # Training data
    └── test_catvnoncat.h5              # Test data
```

## Key Concepts Demonstrated

### Why Deep Networks?

Shallow networks can theoretically represent any function but may require exponentially many neurons. Deep networks achieve efficient representation through:

**Hierarchical Feature Learning:**
- **Layer 1:** Edges, color gradients
- **Layer 2:** Simple shapes, textures
- **Layer 3:** Object parts (ears, eyes)
- **Layer 4:** High-level features → classification

### Cache Mechanism

Forward propagation stores intermediate values for efficient backpropagation:

```python
cache = (linear_cache, activation_cache)
# linear_cache = (A_prev, W, b)
# activation_cache = Z

caches = [cache1, cache2, ..., cacheL]
```

This avoids recomputation during backward pass, reducing complexity from O(L²) to O(L).

### Activation Function Choice

| Layer Type | Activation | Reason |
|------------|------------|--------|
| Hidden | ReLU | Avoids vanishing gradient, sparse activation |
| Output | Sigmoid | Outputs probability in [0, 1] for binary classification |

## Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `learning_rate` | 0.0075 | Gradient descent step size |
| `num_iterations` | 2500 | Training iterations |
| `layer_dims` | [12288, 20, 7, 5, 1] | Network architecture |

### Tuning Guidelines

- **Learning rate too high:** Cost oscillates or diverges
- **Learning rate too low:** Slow convergence
- **Too few layers:** Underfitting, poor feature extraction
- **Too many layers:** Overfitting, vanishing gradients

## Visualization

### Cost Curve

```python
def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
```

### Mislabeled Images

```python
print_mislabeled_images(classes, test_x, test_y, pred_test)
```

Displays images that the model classified incorrectly for error analysis.

## Dependencies

```
numpy>=1.19.0
matplotlib>=3.3.0
h5py>=2.10.0
scipy>=1.5.0
pillow>=7.2.0
```

## Testing

The implementation includes comprehensive unit tests:

```python
# Test parameter initialization
initialize_parameters_test_1(initialize_parameters)
initialize_parameters_deep_test_1(initialize_parameters_deep)

# Test forward propagation
linear_forward_test(linear_forward)
linear_activation_forward_test(linear_activation_forward)
L_model_forward_test(L_model_forward)

# Test backward propagation
linear_backward_test(linear_backward)
linear_activation_backward_test(linear_activation_backward)
L_model_backward_test(L_model_backward)

# Test full models
two_layer_model_test(two_layer_model)
L_layer_model_test(L_layer_model)
```

## Limitations and Extensions

### Current Limitations

- No regularization (L2, dropout)
- Basic gradient descent only (no momentum, Adam)
- Binary classification only
- No batch processing (full batch gradient descent)

### Potential Extensions

- Add L2 regularization to prevent overfitting
- Implement dropout for better generalization
- Add momentum or Adam optimizer
- Support mini-batch gradient descent
- Extend to multi-class with softmax output
- Add batch normalization
- Implement learning rate decay

## References

- [deeplearning.ai Course 1: Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)
- Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016)
- He et al., "Delving Deep into Rectifiers" (2015) - ReLU initialization
