import torch
from torch.nn import MSELoss
from torch.optim import SGD
import matplotlib.pyplot as plt

from utils_data import import_data, VehicleDataset, split_dataset, plot_data
from utils_model import BinaryClassifier, evaluate_model, plot_model


def main():
    # Load and preprocess data
    data_X, data_y = import_data()

    # Create full dataset for training in PyTorch
    full_dataset = VehicleDataset(data_X, data_y)

    # Calculate sizes for train and test sets
    data_loader, _ = split_dataset(full_dataset, ratio=1.0)


    # Initialize model, loss function, and optimizer
    model = BinaryClassifier(
        input_size=data_X.shape[1],  # 2 features: mpg and horsepower
    ) 
    
    # Change the weight of the model in the following section
    #########################################################

    model.weight = torch.tensor([20, -1, 0.])

    #########################################################

    # Evaluate the model on both training and testing data
    accuracy, _ = evaluate_model(model, data_loader)
    print(f'Accuracy of the model on the data: {accuracy:.2f}%')
    
    # Plot the data
    fig, ax = plot_data(data_X, data_y)

    # Plot the decision boundary
    _, ax = plot_model(model, ax)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()