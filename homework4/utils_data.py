from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FONTSIZE = 16

def import_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load data
    link = 'https://raw.githubusercontent.com/Mehta-Research-Group-UIUC/ME360_Fall2024/main/homework4/auto-mpg.csv'
    data = pd.read_csv(link)

    # Data preprocessing
    data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
    data = data.dropna()

    # Filter for only 4 and 8 cylinder vehicles
    data = data[data['cylinders'].isin([4, 8])]

    # Separate 4-cylinder and 8-cylinder vehicles
    data_4cyl = data[data['cylinders'] == 4]
    data_8cyl = data[data['cylinders'] == 8]

    # Find the minimum count
    min_count = min(len(data_4cyl), len(data_8cyl))

    # Balance the dataset
    data_4cyl_balanced = data_4cyl.sample(n=min_count, random_state=42)
    data_8cyl_balanced = data_8cyl.sample(n=min_count, random_state=42)

    # Combine the balanced datasets
    data_balanced = pd.concat([data_4cyl_balanced, data_8cyl_balanced])

    # Shuffle the balanced dataset
    data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Select features and target
    data_X = data_balanced[['mpg', 'horsepower']]
    data_y = data_balanced['cylinders'] 

    return data_X, data_y

class VehicleDataset(Dataset):
    def __init__(self, data_X: pd.DataFrame, data_y: pd.DataFrame):
        self.X = torch.tensor(data_X.values, dtype=torch.float32)
        # Label 4 cylinders as -1, 8 cylinders as 1
        self.y = torch.tensor(data_y.map({4: -1, 8: 1}).values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def split_dataset(
    full_dataset: VehicleDataset, 
    ratio: float = 0.8,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    print_samples: bool = False,
) -> tuple[torch.tensor, Optional[torch.tensor]] :
    assert 0 < ratio <= 1, "ratio must be greater than 0 and no greater than 1"


    if np.isclose(ratio, 1):

        _batch_size = len(full_dataset) if batch_size is None else batch_size

        # Return the full dataset
        return DataLoader(full_dataset, batch_size=_batch_size, shuffle=False), None

    number_of_samples = int(ratio * len(full_dataset))

    if seed is None:
        seed = np.random.randint(0, 1000)
    # Create random sample split
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [number_of_samples, len(full_dataset) - number_of_samples], 
        generator=torch.Generator().manual_seed(seed)
    )

    # Create DataLoaders with splited datasets
    _batch_size = len(train_dataset) if batch_size is None else batch_size
    train_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)

    _batch_size = len(test_dataset) if batch_size is None else batch_size
    test_loader = DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

    if print_samples:
        print(f"Content of {number_of_samples} random samples are shown as follows:")
        for inputs, labels in train_loader:
            print(f"Sample: {inputs=}, {labels=}")
        print("\n")

    return train_loader, test_loader


def plot_data(
    data_X: pd.DataFrame, 
    data_y: pd.DataFrame, 
    highlight_loader: Optional[DataLoader] = None,
    labeled: bool = False,
    plot_legend: bool = True,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    # Define colors for 4 and 8 cylinders
    cylinders_colors_dict = {4: 'C0', 8: 'C1'}

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Create scatter plots for each cylinder count
    for cylinders in cylinders_colors_dict.keys():
        # Only select data for the current cylinder count
        mask = (data_y == cylinders)
        # Plot the data
        ax.scatter(
            data_X.loc[mask, 'mpg'], 
            data_X.loc[mask, 'horsepower'], 
            c=cylinders_colors_dict[cylinders], 
            label=f'{cylinders}-cylinder', 
            alpha=0.8,
        )

    # Highlight the selected samples
    if highlight_loader is not None:
        for inputs, labels in highlight_loader:
            for k in range(len(inputs)):
                # Circle the selected sample
                ax.scatter(
                    inputs[k][0], 
                    inputs[k][1],
                    facecolors='none',
                    edgecolors='green',
                    s=200,
                    linewidth=2,
                    linestyle='--',
                )
                if labeled:
                    # Add text label
                    ax.annotate(
                        f'{labels[k]}', 
                        (inputs[k][0], inputs[k][1]), 
                        xytext=(8, 8), 
                        textcoords='offset points', 
                        fontsize=FONTSIZE-2, 
                        color='green', 
                        fontweight='bold'
                    )
    
    
    if plot_legend:
        ax.legend(fontsize=FONTSIZE)
        ax.set_xlabel('$x_1$ Miles per Gallon (MPG)', fontsize=FONTSIZE)
        ax.set_ylabel('$x_2$ Horsepower (HP)', fontsize=FONTSIZE)
    
    return fig, ax