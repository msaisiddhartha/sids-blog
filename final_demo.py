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
