# Logistic Regression with Neural Network Mindset

## Binary Image Classification from Scratch

## Overview

A logistic regression classifier implemented from scratch using Python and NumPy, designed to classify images as containing a cat (1) or not containing a cat (0). This project frames logistic regression as a single-neuron neural network, building foundational intuition for deep learning concepts including forward propagation, cost computation, backward propagation, and gradient descent optimization.

## Author

- **Anmol Dubey**

## Features

| Feature | Description |
|---------|-------------|
| Vectorized Implementation | Efficient NumPy operations processing all examples simultaneously |
| Complete Training Pipeline | Initialize → Forward → Cost → Backward → Update cycle |
| Image Preprocessing | Flattening and normalization for neural network input |
| Learning Curve Visualization | Cost tracking across iterations |
| Prediction Interface | Binary classification with probability threshold |

## Architecture

Logistic regression as a single-layer neural network:

```
Input Layer                     Output Layer
                                (Sigmoid Activation)
    x₁  ─────┐
    x₂  ─────┤
    x₃  ─────┼────►  Σ(wᵢxᵢ + b)  ────►  σ(z)  ────►  ŷ ∈ [0,1]
    ...      │
    x₁₂₂₈₈ ──┘

         W ∈ ℝ^(12288×1)              Prediction
         b ∈ ℝ                        P(cat | image)
```

**Dimensions:**
- Input: 64×64×3 = 12,288 features per image
- Output: Single probability value

## Mathematical Foundations

### Forward Propagation

```
z = wᵀX + b
A = σ(z) = 1 / (1 + e⁻ᶻ)
```

Where:
- `X` is the input matrix (n_features × m_examples)
- `w` is the weight vector (n_features × 1)
- `b` is the bias scalar
- `A` contains predictions for all examples

### Cost Function

Binary cross-entropy (log loss):

```
J(w, b) = -(1/m) · Σᵢ[y⁽ⁱ⁾log(a⁽ⁱ⁾) + (1-y⁽ⁱ⁾)log(1-a⁽ⁱ⁾)]
```

This measures how well predictions match true labels:
- Correct confident prediction → low cost
- Wrong confident prediction → high cost

### Backward Propagation

Gradient computation for parameter updates:

```
dw = (1/m) · X · (A - Y)ᵀ
db = (1/m) · Σ(A - Y)
```

### Parameter Update

Gradient descent:

```
w := w - α · dw
b := b - α · db
```

Where α is the learning rate.

## Dataset

**Cat vs Non-Cat Image Classification**

| Property | Training Set | Test Set |
|----------|--------------|----------|
| Examples | 209 | 50 |
| Image Size | 64×64×3 | 64×64×3 |
| Features | 12,288 | 12,288 |
| Positive Class | Cat (1) | Cat (1) |
| Negative Class | Non-cat (0) | Non-cat (0) |

### Data Preprocessing Pipeline

```python
# Original shape: (m, 64, 64, 3)
# Step 1: Flatten images
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# Result shape: (12288, m)

# Step 2: Normalize pixel values
train_set_x = train_set_x_flatten / 255.
# Result: values in [0, 1]
```

## Implementation Details

### Core Functions

| Function | Purpose | Signature |
|----------|---------|-----------|
| `sigmoid` | Activation function | `sigmoid(z) → s` |
| `initialize_with_zeros` | Parameter initialization | `initialize_with_zeros(dim) → w, b` |
| `propagate` | Forward + backward pass | `propagate(w, b, X, Y) → grads, cost` |
| `optimize` | Gradient descent loop | `optimize(w, b, X, Y, ...) → params, grads, costs` |
| `predict` | Binary classification | `predict(w, b, X) → Y_prediction` |
| `model` | Complete training pipeline | `model(X_train, Y_train, ...) → d` |

### Function Details

#### Sigmoid Activation

```python
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
```

Maps any real number to (0, 1) for probability interpretation.

#### Propagation (Forward + Backward)

```python
def propagate(w, b, X, Y):
    m = X.shape[1]
    
    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    
    # Backward propagation
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)
    
    grads = {"dw": dw, "db": db}
    return grads, cost
```

#### Prediction

```python
def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = (A > 0.5).astype(int)
    return Y_prediction
```

## Usage

### Running the Model

```bash
python logisticRegression_With_NeuralNetworkMindset.py
```

### Training a Custom Model

```python
from lr_utils import load_dataset

# Load data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Preprocess
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255.
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255.

# Train model
d = model(
    train_set_x, train_set_y,
    test_set_x, test_set_y,
    num_iterations=2000,
    learning_rate=0.005,
    print_cost=True
)
```

### Classifying New Images

```python
# Load and preprocess new image
from PIL import Image
import numpy as np

image = Image.open("my_image.jpg").resize((64, 64))
image_array = np.array(image).reshape(1, -1).T / 255.

# Predict
prediction = predict(d["w"], d["b"], image_array)
print("Cat" if prediction[0,0] == 1 else "Not a cat")
```

## Results

### Training Output

```
Cost after iteration 0: 0.693147
Cost after iteration 100: 0.584508
Cost after iteration 200: 0.466949
...
Cost after iteration 1900: 0.083734

train accuracy: 99.04306220095694 %
test accuracy: 70.0 %
```

### Performance Summary

| Metric | Value |
|--------|-------|
| Training Accuracy | ~99% |
| Test Accuracy | ~70% |
| Training Examples | 209 |
| Test Examples | 50 |

The gap between training and test accuracy indicates some overfitting, which is expected given the small dataset size.

### Learning Curve

The cost decreases smoothly over iterations, indicating successful optimization:

```
Cost
 │
0.7├──●
   │    ╲
0.5├      ╲
   │        ╲
0.3├          ╲
   │            ──●
0.1├                ──────●
   └──────────────────────────► Iterations
   0    500   1000  1500  2000
```

## File Structure

```
logisticRegression-With-NeuralNetworkMindset/
├── README.md                                      # This file
├── logisticRegression_With_NeuralNetworkMindset.py # Main implementation
├── projectDescription                             # Project summary
├── lr_utils.py                                    # Data loading utilities
├── public_tests.py                                # Unit tests
└── datasets/
    ├── train_catvnoncat.h5                        # Training data
    └── test_catvnoncat.h5                         # Test data
```

## Key Concepts Demonstrated

### Neural Network Mindset

This implementation frames logistic regression using neural network terminology:

| Logistic Regression | Neural Network Equivalent |
|---------------------|---------------------------|
| Input features | Input layer activations |
| Weights w | Layer weights W |
| Bias b | Layer bias b |
| Sigmoid output | Output layer with sigmoid activation |
| Cost function | Loss function |
| Gradient descent | Backpropagation + optimization |

### Vectorization

All operations are vectorized for efficiency:

```python
# Non-vectorized (slow)
for i in range(m):
    z[i] = np.dot(w.T, X[:,i]) + b
    a[i] = sigmoid(z[i])

# Vectorized (fast)
Z = np.dot(w.T, X) + b
A = sigmoid(Z)
```

Vectorization eliminates explicit loops, leveraging optimized linear algebra libraries.

### Why Zero Initialization Works

Unlike deep networks, logistic regression can use zero initialization for weights:
- Single layer means no symmetry breaking needed
- Gradient is non-zero due to input features
- Deep networks require random initialization to break symmetry

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_iterations` | 2000 | Training iterations |
| `learning_rate` | 0.005 | Gradient descent step size |

### Learning Rate Analysis

| Learning Rate | Behavior |
|---------------|----------|
| Too high (0.01+) | Cost oscillates, may diverge |
| Optimal (~0.005) | Smooth convergence |
| Too low (0.0001) | Very slow convergence |

## Limitations

- **Linear Decision Boundary:** Cannot capture complex non-linear patterns
- **Small Dataset:** Only 209 training examples leads to overfitting
- **No Regularization:** No L2 penalty on weights
- **Single Feature Representation:** Raw pixels, no learned features

## Comparison with Deep Networks

| Aspect | Logistic Regression | Deep Neural Network |
|--------|---------------------|---------------------|
| Layers | 1 (input → output) | Multiple hidden layers |
| Decision Boundary | Linear | Non-linear |
| Feature Learning | None | Hierarchical |
| Parameters | n + 1 | Millions possible |
| Training Time | Fast | Slower |
| Expressiveness | Limited | High |

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
# Test sigmoid function
sigmoid_test(sigmoid)

# Test initialization
initialize_with_zeros_test_1(initialize_with_zeros)
initialize_with_zeros_test_2(initialize_with_zeros)

# Test propagation
propagate_test(propagate)

# Test optimization
optimize_test(optimize)

# Test prediction
predict_test(predict)

# Test full model
model_test(model)
```

## Next Steps

This project serves as a foundation for understanding:

1. **Shallow Neural Networks:** Add hidden layer with non-linear activation
2. **Deep Neural Networks:** Stack multiple hidden layers
3. **Regularization:** Add L2 penalty to prevent overfitting
4. **Advanced Optimizers:** Momentum, RMSprop, Adam
5. **Convolutional Networks:** Specialized architecture for images

## References

- [deeplearning.ai Course 1, Week 2](https://www.coursera.org/learn/neural-networks-deep-learning)
- Andrew Ng, "Machine Learning" (Stanford CS229)
- Bishop, *Pattern Recognition and Machine Learning* (Chapter 4)
