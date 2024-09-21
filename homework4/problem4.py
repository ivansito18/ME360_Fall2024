import torch
from torch.nn import MSELoss
from torch.optim import SGD
import matplotlib.pyplot as plt

from utils_data import import_data, VehicleDataset, split_dataset, plot_data
from utils_model import BinaryClassifier, evaluate_model, plot_model
from utils_algo import compute_least_squares_solution

def main():
    # Load and preprocess data
    data_X, data_y = import_data()

    # Create full dataset for training in PyTorch
    full_dataset = VehicleDataset(data_X, data_y)

    # Create a data loader for the full dataset
    data_loader, _ = split_dataset(full_dataset, ratio=1.0)


    # Initialize model, loss function, and optimizer
    model = BinaryClassifier(
        input_size=data_X.shape[1],  # 2 features: mpg and horsepower
    ) 
    loss_func = MSELoss()  # Mean Squared Error Loss

    # Compute the least squares solution
    model.weight = compute_least_squares_solution(data_loader)
    print(f'Least squares solution: {model.weight}')

    # Evaluate the model on both training and testing data
    accuracy, loss = evaluate_model(model, data_loader, loss_func)

    print(f'For the model with weight: {model.weight}')
    print(f'Accuracy of the model on the data: {accuracy:.2f}%. Loss: {loss/2:.4f}\n')
    
    # Plot the data
    fig, ax = plot_data(data_X, data_y)

    # Plot the decision boundary
    _, ax = plot_model(model, ax)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()