import torch
from torch.utils.data import DataLoader

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
