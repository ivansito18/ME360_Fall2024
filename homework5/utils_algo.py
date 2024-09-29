import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils_data import plot_data
from utils_model import BinaryClassifier, plot_model

FONTSIZE = 16

def compute_least_squares_solution(
    train_loader: DataLoader, 
) -> torch.Tensor:
    X_list = []
    y_list = []

    for inputs, labels in train_loader:
        # Add a column of ones to inputs for the bias term
        ones = torch.ones(inputs.shape[0], 1)
        data_augmented_X = torch.cat((ones, inputs), dim=1)
        
        X_list.append(data_augmented_X)
        y_list.append(labels.view(-1, 1))

    # Concatenate all batches
    data_augmented_X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    # Compute the least squares solution
    # w = (X^T X)^(-1) X^T y
    X_Transpose = torch.transpose(data_augmented_X, 0, 1)
    w = torch.inverse(X_Transpose @ data_augmented_X) @ X_Transpose @ y

    return w.squeeze()

def gradient_descent(
    model : BinaryClassifier, 
    train_loader: DataLoader, 
    loss_func: nn.Module, 
    learning_rate: float,
    number_of_iterations: int,
) -> list:
    assert learning_rate > 0, 'Learning rate must be greater than 0'
    assert isinstance(learning_rate, float), 'Learning rate must be a float'
    assert number_of_iterations > 0, 'Number of iterations must be greater than 0'
    assert isinstance(number_of_iterations, int), 'Number of iterations must be an integer'
    
    torch.manual_seed(33)
    weight_history = []
    loss_history = []
    for _ in range(number_of_iterations):
        for inputs, labels in train_loader:
            labels = labels.view(-1, 1).float()

            # Forward pass
            outputs = model(inputs)
            loss: torch.Tensor = loss_func(outputs, labels)
            
            # Backward pass
            model.zero_grad()
            loss.backward()

            # Record the weight and loss
            weight_history.append(model.weight.detach().numpy())
            loss_history.append(loss.item())
            
            # update parameters using gradient descent
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

    for inputs, labels in train_loader:
        labels = labels.view(-1, 1).float()
        # Forward pass
        outputs = model(inputs)
        loss: torch.Tensor = loss_func(outputs, labels)

        # Record the final weight and loss
        weight_history.append(model.weight.detach().numpy())
        loss_history.append(loss.item())
      
    
    return weight_history, loss_history

def plot_training(
    loss_history: list, 
    weight_history: list,
    data_X: pd.DataFrame,
    data_y: pd.DataFrame,
    model: BinaryClassifier,
) -> tuple[plt.Figure, list[plt.Axes]]:
    
    fig = plt.figure(figsize=(12, 8)) 
    fig.suptitle('Training Progress', fontsize=FONTSIZE+2)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5])
    axes = []

    # Plot loss history
    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(loss_history)
    ax_loss.set_xlabel('Iteration', fontsize=FONTSIZE)
    ax_loss.set_ylabel('Loss', fontsize=FONTSIZE)
    axes.append(ax_loss)

    # Select iterations for weight plots
    iterations = [0, len(weight_history) // 2, -1]  # Start, middle, end
    titles = ['Initial Weight', 'Mid-training Weight', 'Final Weight']

    # Create weight plots with shared y-axis
    for i, (iteration, title) in enumerate(zip(iterations, titles)):
        ax_weight = fig.add_subplot(gs[1, i])

        # Share y-axis for all weight plots
        if i > 0:
            ax_weight.sharey(axes[1])
        
        weight = weight_history[iteration]
        
        # Plot the data
        _, ax_weight = plot_data(
            data_X, data_y, 
            ax=ax_weight,
            plot_legend=False,
        )
        
        # Plot the decision boundary
        model.weight = torch.tensor(weight)
        _, ax_weight = plot_model(
            model, 
            ax_weight,
            plot_legend=False,
        )

        ith_iteration = len(loss_history) + iteration if iteration < 0 else iteration

        ax_weight.set_title(f'Iteration {ith_iteration}', fontsize=FONTSIZE)
        ax_loss.axvline(ith_iteration, color='grey', linestyle='--', alpha=0.5)

        axes.append(ax_weight)

    return fig, axes