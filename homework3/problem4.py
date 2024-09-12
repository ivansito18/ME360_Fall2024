# !sudo apt-get install texlive-full # Uncomment this line if you are running this code on Google Colab

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

FONTSIZE = 16

def generate_input_signal(time):
    value = torch.zeros_like(time)
    mask = time >= 0
    value[mask] = torch.sin(time[mask]).pow(3)
    return value

def plot_signals(time, signal_input, signal_output):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlabel(r'$t$', fontsize=FONTSIZE)
    ax.grid(True)
    ax.axhline(y=0, color='black', linestyle='-', alpha=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=1)
    ax.plot(time.numpy(), signal_input.numpy(), linewidth=5, label='Input Signal $u(t)$')
    ax.plot(time.numpy(), signal_output.detach().numpy(), linewidth=5, label='Output Signal $y(t)$')
    ax.set_xlim(-1, 11)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE, loc='lower right', bbox_to_anchor=(1, 1), ncol=2)
    plt.show()

class LinearTransformation(nn.Module):
    def __init__(self, weight):
        super().__init__()
        input_dim = weight.shape[1]
        output_dim = weight.shape[0]

        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.update_weight(weight)
    
    def forward(self, x):
        return self.linear(x)
    
    def update_weight(self, weight):
        with torch.no_grad():
            self.linear.weight.copy_(weight)


def main():

    # Part 1: Input Signal Generation
    delta_t = 1./10.
    start_time = -2
    end_time = 12
    start_discrete_time = int(start_time / delta_t)
    end_discrete_time = int(end_time / delta_t)
    discrete_time = torch.arange(start_discrete_time, end_discrete_time+1)
    time = torch.linspace(start_time, end_time, int((end_time - start_time) / delta_t)+1)
    input_signal = generate_input_signal(time)

    # Part 2: Signal Transformation
    
    ## Create a LinearTransformation instance
    linear_transform_weight = torch.tensor([[1., 1.]]) # Define the transformation matrix W
    linear_transform = LinearTransformation(
        weight=linear_transform_weight
    )

    ## Create Output Signal
    output_signal = torch.zeros_like(input_signal)
    past_input_length = linear_transform_weight.shape[1]
    for index, n in enumerate(discrete_time[past_input_length-1:]):
        input_history = input_signal[index: index+past_input_length]
        # The next line means: output_signal = W @ input_history where W is the transformation matrix
        # It calls the forward method of the LinearTransformation instance
        output_signal[past_input_length-1+index] = linear_transform(input_history) 
        

    # Part 3: Plotting
    plot_signals(time, input_signal, output_signal)

if __name__ == "__main__":
    main()
