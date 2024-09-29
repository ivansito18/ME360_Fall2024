import torch
from torch.nn import MSELoss, Tanh
import matplotlib.pyplot as plt

from utils_data import import_data, VehicleDataset, split_dataset, plot_data
from utils_model import BinaryClassifier, evaluate_model, plot_model
from utils_algo import gradient_descent, plot_training


def main():
    # Load and preprocess data
    data_X, data_y = import_data()

    # Create full dataset for training in PyTorch
    full_dataset = VehicleDataset(data_X, data_y)

    # Create a data loader for the full dataset
    data_loader, _ = split_dataset(full_dataset, ratio=1.0)

    # Initialize model, loss function, and optimizer
    model = BinaryClassifier(
        input_size=data_X.shape[1], # 2 features: mpg and horsepower
        activation=Tanh(), # Hyperbolic Tangent Activation Function
    )
    loss_func = MSELoss()  # Mean Squared Error Loss

    # Change the initial weight, learning rate, and number of 
    # iterations of the training process in the following section
    #########################################################

    model.weight = torch.tensor([15.0, -0.5, 0.1])
    learning_rate: float = 0.0001
    number_of_iterations: int = 300

    #########################################################

    # Train the model using gradient descent
    weight_history, loss_history = gradient_descent(
        model, 
        data_loader, 
        loss_func, 
        learning_rate,
        number_of_iterations,
    )
    
    # Evaluate the model on both training and testing data
    accuracy, loss = evaluate_model(model, data_loader, loss_func)
    
    print(f'Accuracy of the trained model on the data: {accuracy:.2f}%. Loss: {loss:.2f}')
    
    # Plot the loss history with the weight at some iterations
    loss_fig, axes = plot_training(loss_history, weight_history, data_X, data_y, model)    
    loss_fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()