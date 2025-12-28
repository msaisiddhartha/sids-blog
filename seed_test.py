import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def run_experiment(seed):
    torch.manual_seed(seed)
    # 1. Ground Truth
    def ground_truth(x): return (x - 2)**2 + 1

    # 2. Data with Gap [1.5, 2.5]
    x_part1 = torch.linspace(-2, 1.5, 50)
    x_part2 = torch.linspace(2.5, 6, 50)
    X_train = torch.cat([x_part1, x_part2]).view(-1, 1)
    y_train = ground_truth(X_train)

    # 3. Model
    model = nn.Sequential(
        nn.Linear(1, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )

    # 4. Train
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(1000):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    # 5. Optimize Input
    param_x = torch.tensor([0.0], requires_grad=True)
    opt_param = optim.Adam([param_x], lr=0.1)
    for _ in range(100):
        opt_param.zero_grad()
        loss = model(param_x.view(1,1))
        loss.backward()
        opt_param.step()
    
    return param_x.item()

print("Testing seeds...")
for seed in [0, 1, 42, 123, 999]:
    res = run_experiment(seed)
    print(f"Seed {seed}: Result {res:.4f}")
