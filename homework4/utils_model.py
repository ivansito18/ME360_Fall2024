from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


FONTSIZE = 16

class BinaryClassifier(nn.Module):
    def __init__(
        self, 
        input_size: Optional[int]= None, 
        activation: nn.Module = nn.Identity(),
        weight: Optional[torch.Tensor] = None,
    ):
        super(BinaryClassifier, self).__init__()
        if weight is not None:
            input_size = weight.shape[0] - 1
        self.linear = nn.Linear(input_size, 1)
        self.activation = activation
        if weight is not None:
            self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return self.activation(y)
    
    @property
    def weight(self) -> torch.Tensor:
        return torch.cat([self.linear.bias, self.linear.weight.squeeze()])

    @weight.setter
    def weight(self, weight: torch.Tensor) -> None:
        with torch.no_grad():
            self.linear.weight.copy_(weight[1:])
            self.linear.bias.copy_(weight[0])

    
def evaluate_model(
    model: nn.Module, 
    data_loader: DataLoader,
    loss_func: Optional[nn.Module] = None,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = None if loss_func is None else 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:

            outputs: torch.Tensor = model(inputs)
            predicted = torch.sign(outputs.squeeze())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if total_loss is not None:
                labels_reshape = labels.view(-1, 1).float()
                loss: torch.Tensor = loss_func(outputs, labels_reshape)
                total_loss += loss.item()
    
    accuracy = 100 * correct / total
    return accuracy, total_loss

def plot_model(model: BinaryClassifier, ax: plt.Axes, plot_legend: bool = True) -> tuple[plt.Figure, plt.Axes]:
    # Create a grid of points to evaluate the model
    mpg_axis_min, mpg_axis_max = ax.get_xlim()
    hp_axis_min, hp_axis_max = ax.get_ylim()
    grid_resolution = 100
    mpg_grid, hp_grid = np.meshgrid(
        np.linspace(mpg_axis_min, mpg_axis_max, grid_resolution), 
        np.linspace(hp_axis_min, hp_axis_max, grid_resolution)
    )
    X_grid = torch.tensor(np.c_[mpg_grid.ravel(), hp_grid.ravel()], dtype=torch.float32)

    # Evaluate the model on the grid
    output_grid = model(X_grid).detach().numpy().reshape(mpg_grid.shape)

    # Plot the decision boundary    
    ax.contourf(mpg_grid, hp_grid, output_grid, alpha=0.3)
    weight = model.weight.detach().numpy()
    if not np.isclose(weight[2], 0):
        x = np.linspace(mpg_axis_min, mpg_axis_max, 100)
        y = - (weight[0] + weight[1] * x) / weight[2]
        ax.plot(x, y, color='red', linestyle='--', label='Decision Boundary')
    elif not np.isclose(weight[1], 0):
        x = -weight[0] / weight[1]
        ax.axvline(x, color='red', linestyle='--', label='Decision Boundary')
    else:
        pass
    
    ax.set_xlim(mpg_axis_min, mpg_axis_max)
    ax.set_ylim(hp_axis_min, hp_axis_max)

    if plot_legend:
        ax.legend(fontsize=FONTSIZE)
    
    fig = ax.get_figure()

    return fig, ax
