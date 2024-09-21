import matplotlib.pyplot as plt

from utils_data import (
    import_data, 
    VehicleDataset, 
    split_dataset,
    plot_data,
)


def main():
    # Load and preprocess data
    data_X, data_y = import_data()
    print(f"There are {len(data_X)} samples in the dataset.")
    print(f"There are {data_X.shape[1]} features in the input of dataset.")
    print(f"Features are: {data_X.columns.values}")
    print(f"There are {len(data_y.unique())} unique labels in the dataset.")
    for label in data_y.unique():
        print(f"Dataset with {label}-cylinder engines has {len(data_y[data_y == label])} samples.")


    # Create full dataset for training in PyTorch
    full_dataset = VehicleDataset(data_X, data_y)

    # Show random samples
    highlight_loader, _ = split_dataset(
        full_dataset, 
        ratio=0.05, 
        batch_size=1,
        print_samples=True,
    )

    # Plot the data
    fig, ax = plot_data(data_X, data_y, highlight_loader, labeled=True)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()