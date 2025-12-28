---
title: "Parametric Optimization using Neural Networks as Surrogates"
date: 2025-12-27T14:15:00-05:00
draft: false
tags: ["Machine Learning", "Optimization", "Neural Networks", "PyTorch"]
categories: ["Tech"]
math: true
ShowToc: true
TocOpen: true

---

Start using Neural Networks not just for prediction, but for **optimization**.

In many engineering and scientific problems, we often deal with "black box" functions where the relationship between inputs and outputs is complex or expensive to compute (e.g., fluid dynamics simulations, finite element analysis). We want to find the input parameters that minimize (or maximize) some objective, but we can't easily calculate gradients to guide us.

This is where **Neural Networks as Surrogate Models** come in.

## The Concept

1.  **Surrogate Modeling**: Train a Neural Network to approximate your expensive or unknown function $f(x)$ based on a set of sampled data points.
    > **Why this works:** The **Universal Approximation Theorem** guarantees that a feed-forward network with even a single hidden layer (and enough neurons) can approximate any continuous function to an arbitrary degree of accuracy. This massive theoretical flexibility is what allows us to confidently swap out complex physics simulations for a Neural Network.
2.  **Differentiability**: Unlike the black box function, a Neural Network is fully differentiable.
3.  **Optimization**: Once trained, we freeze the network's weights. We then treat the *input* $x$ as the trainable parameter. By backpropagating the loss purely with respect to the input, we can use gradient descent to find the optimal $x$ that minimizes the output $y$.

## Example: Minimizing a Parabola

Let's demonstrate this with a trivial example so the mechanics are clear.

Suppose we want to minimize the function:
$$ y = (x - 2)^2 + 1 $$

We know the minimum is at $x=2$ where $y=1$. But let's pretend we don't know the analytical form. We only have data samples.

### Generalization Test
To make this more realistic, we will:
1.  **Split Data**: create a training set and a testing set.
2.  **Create a Gap**: Explicitly **exclude** the region around the optimum ($x \in [1.5, 2.5]$) from the training data. This forces the Neural Network to *interpolate* the shape of the function to find the minimum, rather than just memorizing a data point near $x=2$.

![Optimization Landscape](/images/parabola_optimization.png)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. The Ground Truth Function (The "Black Box")
def ground_truth(x):
    return (x - 2)**2 + 1

# 2. Generate Data with a GAP at the optimum
# Optimum is at x=2. We exclude [1.5, 2.5] from training to test generalization.
# Training data: [-2, 1.5] U [2.5, 6]
x_part1 = torch.linspace(-2, 1.5, 100)
x_part2 = torch.linspace(2.5, 6, 100)
X_train = torch.cat([x_part1, x_part2]).view(-1, 1)
y_train = ground_truth(X_train)

print(f"Training samples: {len(X_train)}")
print(f"The optimum (x=2) is explicitly REMOVED from training data.")

# 3. Define the Surrogate Model
class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(), # Tanh often approximates smooth curves better than ReLU
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

model = SurrogateModel()

# 4. Train the Surrogate
optimizer_model = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Training Surrogate Model...")
for epoch in range(1000):
    optimizer_model.zero_grad()
    prediction = model(X_train)
    loss = criterion(prediction, y_train)
    loss.backward()
    optimizer_model.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.5f}")

print("Surrogate trained!\n")

# 5. Parametric Optimization
# We want to find 'x' that minimizes the output of our trained model.
# Start with a random guess far from the optimum, e.g., x = 0.0
param_x = torch.tensor([0.0], requires_grad=True)

# Optimizer for the input parameter
optimizer_param = optim.Adam([param_x], lr=0.1)

print(f"Initial Guess: x = {param_x.item()}")
print("Optimizing 'x' to find minimum...")

for step in range(100):
    optimizer_param.zero_grad()
    
    # Pass our parameter through the FROZEN surrogate
    pred_y = model(param_x.view(1, 1))
    
    # We want to minimize y, so 'loss' is just the output y itself
    loss_val = pred_y 
    
    loss_val.backward()
    optimizer_param.step()
    
    if step % 20 == 0:
        print(f"Step {step}: x_est = {param_x.item():.4f} -> predicted_y = {pred_y.item():.4f}")

print(f"\nFinal Result:")
print(f"Estimated x_min: {param_x.item():.4f}")
print(f"True x_min: 2.0000")
```

Even though the network never saw data points between $1.5$ and $2.5$, it successfully learned the parabolic curve and guided the optimization to the correct minimum at $x \approx 2.0$.

> **Note**: You might see a result like `1.98` or `2.05` instead of exactly `2.0`. This difference arises because the model is **interpolating** across the gap. Theoretically, infinite curves could fit the training data; the Neural Network picks the "smoothest" one, which is usually very close to, but not always exactly, the true function in the missing region.

---

## Multiple Dimensions: Rosenbrock Function

Let's try something harder: the **Rosenbrock function** (often called the "banana function"). It's a non-convex function used as a performance test problem for optimization algorithms.

$$ f(x, y) = (1 - x)^2 + 100(y - x^2)^2 $$

The global minimum is inside a long, narrow, parabolic shaped flat valley at $(x, y) = (1, 1)$. To find it, our Neural Network must navigate this valley.

![Rosenbrock Contour](/images/rosenbrock_contour.png)

### Normalization is Key
For functions with large values (the Rosenbrock term $100(y-x^2)^2$ explodes easily), **input/output normalization** is crucial for the Neural Network to learn effectively.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Rosenbrock Function
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# 2. Generate Data
num_points = 5000
x_vals = torch.rand(num_points, 1) * 4 - 2 # [-2, 2]
y_vals = torch.rand(num_points, 1) * 4 - 1 # [-1, 3]
inputs = torch.cat([x_vals, y_vals], dim=1)
targets = rosenbrock(inputs[:, 0], inputs[:, 1]).view(-1, 1)

# Standardize Data (Important for Convergence!)
input_mean, input_std = inputs.mean(dim=0), inputs.std(dim=0)
target_mean, target_std = targets.mean(), targets.std()

inputs_norm = (inputs - input_mean) / input_std
targets_norm = (targets - target_mean) / target_std

# 3. Model
model = nn.Sequential(
    nn.Linear(2, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 1)
)

# 4. Train
optimizer_model = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("Training...")
for epoch in range(2000):
    optimizer_model.zero_grad()
    loss = loss_fn(model(inputs_norm), targets_norm)
    loss.backward()
    optimizer_model.step()

# 5. Optimize Input
# Start at random point (0, 3)
start_norm = (torch.tensor([0.0, 3.0]) - input_mean) / input_std
param_norm = start_norm.clone().detach().requires_grad_(True)
opt_params = optim.Adam([param_norm], lr=0.05)

print("Optimizing...")
for step in range(200):
    opt_params.zero_grad()
    # Minimize the normalized output
    model(param_norm.unsqueeze(0)).backward()
    opt_params.step()

# Denormalize
final_res = param_norm * input_std + input_mean
print(f"Found Minimum: ({final_res[0].item():.4f}, {final_res[1].item():.4f})")
print(f"True Minimum:  (1.0000, 1.0000)")
```

This demonstrates that the generic "Surrogate + Gradient" method scales to multi-variable problems, provided you handle data scaling correctly.

## Alternative Methods: Why Not Random Forests?

You might ask: *Why not use a Random Forest or XGBoost? They are often state-of-the-art for regression.*

We tried it. Here is what happens when you train a Random Forest on the same parabola data and try to find the minimum:

![Random Forest Optimization](/images/rf_opt_comparison.png)

1.  **Blocky Landscape**: Tree-based models output "steps". The function is not smooth.
2.  **Gradient Descent Fails**: Gradients are zero on the flat steps and undefined at the jumps. You simply cannot filter derivatives through a tree.
3.  **Derivative-Free Methods?**: 
    *   **Nelder-Mead** (local search) failed in our test, getting stuck in a local flat region ($x \approx 0$).
    *   **Genetic Algorithms** (Differential Evolution) *did* find the minimum ($x \approx 2.03$), but it required **77 function evaluations** for this simple 1D problem.

**The Verdict**: While Genetic Algorithms *can* optimize non-differentiable surrogates (like Random Forests), they suffer from the **Curse of Dimensionality**. For a 100-dimensional engineering problem, a Genetic Algorithm might need $100,000+$ evaluations. A Neural Network with gradients can solve it in a fraction of the steps.

## Conclusion

We explored two examples of using Neural Networks as differentiable surrogate models for optimization:

1.  **1D Parabola**: A simple convex problem where the network successfully interpolated across a data gap. We showed that using **Tanh activation** provides a smoother, more robust approximation than ReLU without needing careful seed selection.
2.  **2D Rosenbrock**: A complex non-convex problem. We demonstrated that **data normalization (standard scaling)** is critical for convergence when inputs and targets have vastly different magnitudes.
3.  **Comparisons**: We showed that traditional regression methods like **Random Forests fail** at this task because they produce piecewise constant "steps" with zero gradients. While Derivative-Free methods (Genetic Algorithms) can optimize these trees, they require significantly more function evaluations (77 vs. instantaneous) compared to the efficient gradient-based guidance provided by a Neural Network.

By treating the Neural Network not just as a predictor, but as a fully differentiable equation, we unlock the power of gradient-based optimization for any "black box" problem.

### Summary
Using Neural Networks for parametric optimization is a powerful technique when:
1.  **Gradients are unavailable**: The original system is a "black box" simulation.
2.  **Function evaluation is expensive**: You can train the surrogate once and run optimization on it instantly.
3.  **High-dimensional design space**: NNs scale reasonably well to higher dimensions compared to grid search.

By combining the **Universal Approximation** capabilities of Neural Networks with the **Automatic Differentiation** of frameworks like PyTorch, we can solve complex engineering design problems that were previously intractable.
