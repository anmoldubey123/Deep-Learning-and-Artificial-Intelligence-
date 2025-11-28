# Planar Data Classification with One Hidden Layer

## Shallow Neural Network for Non-Linear Decision Boundaries

## Overview

A two-layer neural network implemented from scratch using Python and NumPy to solve a non-linear binary classification problem on planar (2D) data. This project demonstrates the fundamental advantage of neural networks over linear classifiers: the ability to learn complex, curved decision boundaries through hidden layer representations.

## Author

- **Anmol Dubey**

## Features

| Feature | Description |
|---------|-------------|
| Non-Linear Classification | Learns curved decision boundaries impossible for linear models |
| Tanh Hidden Layer | Smooth activation enabling gradient flow |
| Sigmoid Output | Probability output for binary classification |
| Decision Boundary Visualization | Visual comparison with logistic regression |
| Configurable Hidden Size | Experiment with different network capacities |

## The Problem: Why Linear Classifiers Fail

The planar dataset forms a "flower" pattern where classes interleave in a way that no straight line can separate:

```
        Class 1 (Blue)              Logistic Regression
        Class 0 (Red)               Decision Boundary
              
           ●  ○  ●                        ○ | ●
          ○ ●  ● ○                       ○  |  ●
         ●  ○  ○  ●                     ○ ○ | ● ●
          ○ ●  ● ○                       ○  |  ●
           ●  ○  ●                        ○ | ●
           
      Actual Data                    ~47% Accuracy
      (Non-linearly separable)       (Fails completely)
```

**Logistic regression achieves only ~47% accuracy** — barely better than random guessing.

## The Solution: Hidden Layer Representation

A neural network with one hidden layer transforms the input space into a new representation where classes become separable:

```
Input Space (2D)          Hidden Space (4D)           Output
     x₁, x₂         →     h₁, h₂, h₃, h₄      →        ŷ
                    
  ●○●                    Transformed                  ●●●
 ○●●○        W1, b1      features where      W2, b2   ○○○
●○○●        ────────→    classes are        ────────→
 ○●●○        tanh        linearly            sigmoid
  ●○●                    separable
```

**The neural network achieves ~90% accuracy** by learning a non-linear transformation.

## Architecture

```
Input Layer (2)    Hidden Layer (4)     Output Layer (1)
                   (Tanh Activation)    (Sigmoid Activation)

    x₁  ──────┐    ┌─── h₁ ───┐
              ├────┼─── h₂ ───┼─────── ŷ
    x₂  ──────┘    ├─── h₃ ───┤
                   └─── h₄ ───┘

              W1 (4×2)      W2 (1×4)
              b1 (4×1)      b2 (1×1)
```

**Layer Dimensions:**
- Input: 2 features (x, y coordinates)
- Hidden: 4 neurons with tanh activation
- Output: 1 neuron with sigmoid activation

## Mathematical Foundations

### Forward Propagation

**Hidden Layer:**
```
Z1 = W1 · X + b1
A1 = tanh(Z1)
```

**Output Layer:**
```
Z2 = W2 · A1 + b2
A2 = σ(Z2) = 1 / (1 + e^(-Z2))
```

### Cost Function

Binary cross-entropy:
```
J = -(1/m) · Σ[y·log(A2) + (1-y)·log(1-A2)]
```

### Backward Propagation

**Output Layer Gradients:**
```
dZ2 = A2 - Y
dW2 = (1/m) · dZ2 · A1ᵀ
db2 = (1/m) · Σ(dZ2)
```

**Hidden Layer Gradients:**
```
dZ1 = W2ᵀ · dZ2 ⊙ (1 - A1²)    # tanh derivative
dW1 = (1/m) · dZ1 · Xᵀ
db1 = (1/m) · Σ(dZ1)
```

Note: `(1 - A1²)` is the derivative of tanh, since `d/dz[tanh(z)] = 1 - tanh²(z)`.

### Parameter Update

```
W1 := W1 - α · dW1
b1 := b1 - α · db1
W2 := W2 - α · dW2
b2 := b2 - α · db2
```

## Dataset

**Planar Flower Dataset**

| Property | Value |
|----------|-------|
| Examples | 400 |
| Features | 2 (x, y coordinates) |
| Classes | 2 (red: 0, blue: 1) |
| Pattern | Flower/spiral shape |
| Separability | Non-linear only |

### Data Visualization

```python
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
```

## Implementation Details

### Core Functions

| Function | Purpose | Signature |
|----------|---------|-----------|
| `layer_sizes` | Define network architecture | `layer_sizes(X, Y) → (n_x, n_h, n_y)` |
| `initialize_parameters` | Random weight initialization | `initialize_parameters(n_x, n_h, n_y) → parameters` |
| `forward_propagation` | Compute predictions | `forward_propagation(X, parameters) → A2, cache` |
| `compute_cost` | Cross-entropy loss | `compute_cost(A2, Y) → cost` |
| `backward_propagation` | Compute gradients | `backward_propagation(parameters, cache, X, Y) → grads` |
| `update_parameters` | Gradient descent step | `update_parameters(parameters, grads, learning_rate) → parameters` |
| `nn_model` | Complete training loop | `nn_model(X, Y, n_h, num_iterations, print_cost) → parameters` |
| `predict` | Binary classification | `predict(parameters, X) → predictions` |

### Key Implementation Details

#### Weight Initialization

```python
W1 = np.random.randn(n_h, n_x) * 0.01
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(n_y, n_h) * 0.01
b2 = np.zeros((n_y, 1))
```

Small random weights prevent saturation; zeros would cause symmetry problems.

#### Tanh Activation

```python
A1 = np.tanh(Z1)
```

Properties:
- Output range: (-1, 1)
- Zero-centered (unlike sigmoid)
- Smooth gradients for optimization

#### Cache for Backpropagation

```python
cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
```

Storing intermediate values avoids recomputation during backward pass.

## Usage

### Running the Model

```bash
python planarDataClassification_With_OneHiddenLayer.py
```

### Training with Custom Parameters

```python
from planar_utils import load_planar_dataset

# Load data
X, Y = load_planar_dataset()

# Train model
parameters = nn_model(
    X, Y,
    n_h=4,                    # Hidden layer size
    num_iterations=10000,     # Training iterations
    print_cost=True           # Print progress
)

# Evaluate
predictions = predict(parameters, X)
accuracy = np.mean(predictions == Y) * 100
print(f"Accuracy: {accuracy}%")
```

### Visualizing Decision Boundary

```python
from planar_utils import plot_decision_boundary

plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size 4")
```

## Results

### Training Progress

```
Cost after iteration 0: 0.693048
Cost after iteration 1000: 0.288083
Cost after iteration 2000: 0.254385
Cost after iteration 3000: 0.233864
...
Cost after iteration 9000: 0.218607
```

### Performance Comparison

| Model | Architecture | Accuracy |
|-------|--------------|----------|
| Logistic Regression | Linear | ~47% |
| Neural Network (n_h=1) | 2→1→1 | ~67% |
| Neural Network (n_h=4) | 2→4→1 | ~90% |
| Neural Network (n_h=20) | 2→20→1 | ~90% |

### Decision Boundary Visualization

```
Logistic Regression              Neural Network (n_h=4)
                                 
    ─────────────────              ╭─────────╮
   │ ○ ○ ○ │ ● ● ● │             ╭─┤ ● ● ● ●├─╮
   │ ○ ○ ○ │ ● ● ● │            ╭┤ ○ ○ ○ ○ ○ ├╮
   │ ○ ○ ○ │ ● ● ● │            │ ○ ● ● ● ● ○ │
   │ ○ ○ ○ │ ● ● ● │            ╰┤ ○ ○ ○ ○ ○ ├╯
    ─────────────────              ╰─┤ ● ● ● ●├─╯
                                     ╰─────────╯
   Linear boundary               Curved boundary
   (Wrong!)                      (Correct!)
```

## File Structure

```
planarDataClassification-With-OneHiddenLayer/
├── README.md                                        # This file
├── planarDataClassification_With_OneHiddenLayer.py  # Main implementation
├── projectDescription                               # Project summary
├── planar_utils.py                                  # Utilities
│   ├── load_planar_dataset()
│   ├── plot_decision_boundary()
│   └── sigmoid()
├── testCases_v2.py                                  # Test cases
└── public_tests.py                                  # Unit tests
```

## Key Concepts Demonstrated

### Universal Approximation Theorem

A neural network with a single hidden layer can approximate any continuous function, given enough hidden units. This project demonstrates this practically: the network learns to approximate the complex flower-shaped decision boundary.

### Representation Learning

The hidden layer learns a new representation of the input where classes become linearly separable:

```
Original Space:          Hidden Space:
(x₁, x₂)                 (h₁, h₂, h₃, h₄)

Interleaved classes  →   Separable classes
```

### Why Tanh Over Sigmoid for Hidden Layers

| Property | Tanh | Sigmoid |
|----------|------|---------|
| Output Range | (-1, 1) | (0, 1) |
| Zero-Centered | Yes | No |
| Gradient at 0 | 1.0 | 0.25 |
| Saturation | Both extremes | Both extremes |

Tanh's zero-centered output helps subsequent layers learn more efficiently.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_h` | 4 | Hidden layer neurons |
| `num_iterations` | 10000 | Training iterations |
| `learning_rate` | 1.2 | Gradient descent step size |

### Effect of Hidden Layer Size

| n_h | Capacity | Risk |
|-----|----------|------|
| 1-2 | Low | Underfitting |
| 4-5 | Medium | Good balance |
| 20+ | High | Overfitting possible |

## Comparison: Linear vs Neural Network

### Decision Boundary Expressiveness

```
Linear Model:           Neural Network:
y = wx + b              y = σ(W2 · tanh(W1·x + b1) + b2)

Can only learn:         Can learn:
- Straight lines        - Curves
- Hyperplanes           - Circles
- Linear boundaries     - Complex shapes
```

### Parameter Count

| Model | Parameters | Flexibility |
|-------|------------|-------------|
| Logistic Regression | 2 + 1 = 3 | Very limited |
| NN (n_h=4) | (2×4+4) + (4×1+1) = 17 | Moderate |
| NN (n_h=20) | (2×20+20) + (20×1+1) = 81 | High |

## Dependencies

```
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.23.0
```

## Testing

The implementation includes unit tests for each function:

```python
# Test layer sizes
layer_sizes_test(layer_sizes)

# Test initialization
initialize_parameters_test(initialize_parameters)

# Test forward propagation
forward_propagation_test(forward_propagation)

# Test cost computation
compute_cost_test(compute_cost)

# Test backward propagation
backward_propagation_test(backward_propagation)

# Test parameter updates
update_parameters_test(update_parameters)

# Test full model
nn_model_test(nn_model)

# Test prediction
predict_test(predict)
```

## Limitations and Extensions

### Current Limitations

- Fixed learning rate (no decay)
- No regularization
- Basic gradient descent (no momentum)
- Single hidden layer

### Potential Extensions

- Add multiple hidden layers (deep network)
- Implement L2 regularization
- Add dropout for regularization
- Use ReLU activation
- Implement mini-batch gradient descent
- Add learning rate scheduling
- Experiment with different optimizers (Adam, RMSprop)

## Next Steps

This project bridges logistic regression and deep networks:

1. **Previous:** Logistic Regression (linear boundaries)
2. **Current:** One Hidden Layer (curved boundaries)
3. **Next:** Deep Networks (hierarchical features)

## References

- [deeplearning.ai Course 1, Week 3](https://www.coursera.org/learn/neural-networks-deep-learning)
- Cybenko, "Approximation by Superpositions of a Sigmoidal Function" (1989)
- Hornik, "Multilayer Feedforward Networks are Universal Approximators" (1989)
