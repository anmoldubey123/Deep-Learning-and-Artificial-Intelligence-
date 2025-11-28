# Deep Learning From Scratch

## Neural Network Implementations in Python

## Overview

A collection of neural network projects implemented from scratch using Python and NumPy, progressing from fundamental logistic regression to deep L-layer networks. Each project builds upon the previous, demonstrating core deep learning concepts without relying on high-level frameworks like TensorFlow or PyTorch.

## Author

- **Anmol Dubey**

## Project Summary

| Project | Directory | Concepts | Architecture |
|---------|-----------|----------|--------------|
| Logistic Regression | `logisticRegression-With-NeuralNetworkMindset` | Binary classification, gradient descent, vectorization | Single neuron (no hidden layers) |
| Planar Data Classification | `planarDataClassification-With-OneHiddenLayer` | Non-linear boundaries, tanh activation, shallow networks | 2-layer NN (1 hidden layer) |
| Deep Neural Network | `deepNeuralNetworkApplication` | L-layer networks, ReLU/sigmoid, forward/backward propagation | Configurable depth |

## Learning Progression

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Deep Learning Fundamentals Curriculum                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Level 1: Logistic Regression (Neural Network Mindset)                      │
│  ├── Sigmoid activation function                                            │
│  ├── Binary cross-entropy loss                                              │
│  ├── Gradient descent optimization                                          │
│  ├── Vectorized forward/backward propagation                                │
│  └── Image classification (cat vs non-cat)                                  │
│                                                                             │
│  Level 2: Shallow Neural Network (One Hidden Layer)                         │
│  ├── Multi-neuron hidden layer                                              │
│  ├── Tanh activation for hidden units                                       │
│  ├── Non-linear decision boundaries                                         │
│  ├── Planar data classification                                             │
│  └── Comparison with linear classifiers                                     │
│                                                                             │
│  Level 3: Deep Neural Network (L Layers)                                    │
│  ├── Arbitrary depth architecture                                           │
│  ├── ReLU activation for hidden layers                                      │
│  ├── He/Xavier initialization                                               │
│  ├── Modular forward/backward implementation                                │
│  └── Scalable to complex image recognition                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Concepts Implemented

### Forward Propagation

Each layer computes linear transformation followed by non-linear activation:

```
Z[l] = W[l] · A[l-1] + b[l]
A[l] = g(Z[l])
```

Where `g` is the activation function (sigmoid, tanh, or ReLU).

### Backward Propagation

Gradients computed via chain rule, propagating error from output to input:

```
dZ[l] = dA[l] * g'(Z[l])
dW[l] = (1/m) · dZ[l] · A[l-1].T
db[l] = (1/m) · sum(dZ[l])
dA[l-1] = W[l].T · dZ[l]
```

### Cost Function

Binary cross-entropy for classification tasks:

```
J = -(1/m) · Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### Parameter Update

Gradient descent with configurable learning rate:

```
W[l] = W[l] - α · dW[l]
b[l] = b[l] - α · db[l]
```

## Activation Functions

| Function | Formula | Derivative | Use Case |
|----------|---------|------------|----------|
| Sigmoid | σ(z) = 1/(1+e⁻ᶻ) | σ(z)(1-σ(z)) | Output layer (binary) |
| Tanh | tanh(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | 1-tanh²(z) | Hidden layers (shallow) |
| ReLU | max(0, z) | 1 if z>0 else 0 | Hidden layers (deep) |

## Repository Structure

```
deepLearningFromScratch/
├── README.md                                           # This file
│
├── logisticRegression-With-NeuralNetworkMindset/      # Project 1
│   ├── logisticRegression_With_NeuralNetworkMindset.py
│   ├── projectDescription
│   ├── lr_utils.py                                    # Data loading utilities
│   ├── public_tests.py                                # Unit tests
│   └── datasets/                                      # Cat/non-cat images
│
├── planarDataClassification-With-OneHiddenLayer/      # Project 2
│   ├── planarDataClassification_With_OneHiddenLayer.py
│   ├── projectDescription
│   ├── planar_utils.py                                # Visualization utilities
│   ├── testCases_v2.py                                # Test cases
│   └── public_tests.py                                # Unit tests
│
└── deepNeuralNetworkApplication/                      # Project 3
    ├── deepNeuralNetworkApplication.py
    ├── projectDescription
    ├── dnn_utils.py                                   # Activation functions
    ├── testCases.py                                   # Test cases
    ├── public_tests.py                                # Unit tests
    └── datasets/                                      # Training data
```

## Requirements

```
numpy>=1.19.0
matplotlib>=3.3.0
h5py>=2.10.0
scipy>=1.5.0
scikit-learn>=0.23.0
pillow>=7.2.0
```

### Installation

```bash
pip install numpy matplotlib h5py scipy scikit-learn pillow
```

## Quick Start

```bash
# Run logistic regression classifier
cd logisticRegression-With-NeuralNetworkMindset
python logisticRegression_With_NeuralNetworkMindset.py

# Run shallow neural network
cd ../planarDataClassification-With-OneHiddenLayer
python planarDataClassification_With_OneHiddenLayer.py

# Run deep neural network
cd ../deepNeuralNetworkApplication
python deepNeuralNetworkApplication.py
```

## Results Summary

| Project | Dataset | Training Accuracy | Test Accuracy |
|---------|---------|-------------------|---------------|
| Logistic Regression | Cat vs Non-cat (209 images) | ~99% | ~70% |
| Shallow NN | Planar flower data (400 points) | ~90% | N/A |
| Deep NN (4-layer) | Cat vs Non-cat (209 images) | ~98% | ~80% |

## Key Implementation Functions

### Shared Across Projects

| Function | Purpose |
|----------|---------|
| `initialize_parameters()` | Random weight initialization with small values |
| `forward_propagation()` | Compute activations layer by layer |
| `compute_cost()` | Binary cross-entropy loss calculation |
| `backward_propagation()` | Gradient computation via chain rule |
| `update_parameters()` | Gradient descent parameter update |
| `predict()` | Binary classification from learned parameters |

### Deep Network Specific

| Function | Purpose |
|----------|---------|
| `initialize_parameters_deep()` | L-layer weight initialization |
| `L_model_forward()` | [LINEAR→RELU]*(L-1)→LINEAR→SIGMOID |
| `L_model_backward()` | Full network gradient computation |
| `linear_activation_forward()` | Single layer forward with activation |
| `linear_activation_backward()` | Single layer backward with activation |

## Architecture Diagrams

### Logistic Regression (Project 1)
```
Input (12288)  →  [σ]  →  Output (1)
   X                        ŷ
```

### Shallow Network (Project 2)
```
Input (2)  →  [tanh] (4 units)  →  [σ]  →  Output (1)
   X              A1                         ŷ
```

### Deep Network (Project 3)
```
Input (12288) → [ReLU](20) → [ReLU](7) → [ReLU](5) → [σ] → Output (1)
     X            A1           A2          A3              ŷ
```

## Mathematical Foundations

### Why Deep Networks?

Shallow networks can represent any function but may require exponentially many neurons. Deep networks achieve the same with polynomial complexity through hierarchical feature learning:

- **Layer 1:** Edges, simple patterns
- **Layer 2:** Parts, textures
- **Layer 3:** Objects, complex features
- **Output:** Classification decision

### Vanishing Gradient Problem

Deep networks with sigmoid/tanh suffer from vanishing gradients. ReLU activation mitigates this:

```
ReLU gradient = 1 for z > 0 (no saturation)
Sigmoid gradient ≤ 0.25 (saturates at extremes)
```

## Learning Resources

These implementations follow the deeplearning.ai curriculum structure, providing hands-on experience with:

- Vectorized NumPy operations
- Computational graph thinking
- Gradient checking techniques
- Hyperparameter tuning intuition

## Future Extensions

Potential enhancements to explore:

- Regularization (L2, dropout)
- Batch normalization
- Adam/RMSprop optimizers
- Mini-batch gradient descent
- Learning rate scheduling
- Multi-class classification (softmax)

## References

- [deeplearning.ai Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning)
- Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press)
- Andrew Ng, *Machine Learning Yearning*

---

*These implementations demonstrate fundamental deep learning concepts through from-scratch coding, building intuition before transitioning to production frameworks.*
