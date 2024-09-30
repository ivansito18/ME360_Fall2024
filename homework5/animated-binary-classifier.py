import torch
from torch.nn import MSELoss, Tanh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

from utils_data import import_data, VehicleDataset, split_dataset, plot_data
from utils_model import BinaryClassifier, evaluate_model, plot_model
from utils_algo import gradient_descent

def animate(frame, model, data_X, data_y, weight_history, loss_history, ax_model, ax_loss):
    ax_model.clear()
    ax_loss.clear()
    
    # Plot the data and decision boundary
    plot_data(data_X, data_y, ax=ax_model, plot_legend=False)
    model.weight = torch.tensor(weight_history[frame])
    plot_model(model, ax_model, plot_legend=False)
    
    ax_model.set_title(f'Iteration {frame}')
    ax_model.set_xlabel('Miles per Gallon (MPG)')
    ax_model.set_ylabel('Horsepower (HP)')

    # Plot the loss function
    max_loss = max(loss_history)
    min_loss = min(loss_history)

    ax_loss.plot(loss_history[:frame+1], color='red')
    ax_loss.set_title('Loss Over Time')
    ax_loss.set_ylim(min_loss-(max_loss-min_loss)*0.1, max_loss+(max_loss-min_loss)*0.1)
    ax_loss.set_xlabel('Iteration')
    # ax_loss.set_yscale('log')  # Use log scale for better visualization

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
        activation=Tanh(),  # Hyperbolic Tangent Activation Function
    )
    loss_func = MSELoss()  # Mean Squared Error Loss

    # Set training parameters
    model.weight = torch.tensor([15.0, -0.5, 0.1])
    learning_rate = 0.0001
    number_of_iterations = 300

    # Train the model using gradient descent
    weight_history, loss_history = gradient_descent(
        model, 
        data_loader, 
        loss_func, 
        learning_rate,
        number_of_iterations,
    )
    
    # Evaluate the model on the data
    accuracy, loss = evaluate_model(model, data_loader, loss_func)
    print(f'Accuracy of the trained model on the data: {accuracy:.2f}%. Loss: {loss:.2f}')
    
    # Create the animation
    fig, (ax_model, ax_loss) = plt.subplots(1, 2, figsize=(18, 6))
    ani = FuncAnimation(fig, animate, frames=len(weight_history),
                        fargs=(model, data_X, data_y, weight_history, loss_history, ax_model, ax_loss),
                        interval=50, repeat=False)
    
    # plt.tight_layout()
    # plt.show()
    # Set up the writer for MP4 export
    Writer = writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    # Save the animation as an MP4 file
    ani.save('binary_classifier_training.mp4', writer=writer)
    
    print("Animation saved as 'binary_classifier_training.mp4'")

if __name__ == '__main__':
    main()
