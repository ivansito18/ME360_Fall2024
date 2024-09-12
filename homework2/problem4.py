# !sudo apt-get install texlive-full
from typing import Dict
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
plt.rcParams.update({"text.usetex": True})

def plot_vectors(ax: Axes, fig_name: str, vectors: Dict[str, NDArray]) -> None:
    
    ax.set_title(fig_name)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(y=0, color='k', linestyle='--')
    ax.axvline(x=0, color='k', linestyle='--')
    ax.grid(True, linestyle=':', alpha=0.7)

    # Plot vectors
    for i, (key, vector) in enumerate(vectors.items()):
        ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=f"C{i}", label="$"+key+"$")

    ax.set_aspect('equal')
    ax.legend()
    

def main() -> None:

    fig, (ax_original, ax_transformed) = plt.subplots(1, 2, figsize=(12, 5))

    # Define vectors
    vectors = {
        'v_a': np.array([-2., 1.]),
        'v_b': np.array([1., 1.]),
        'v_c': np.array([0., -2.]),
    }
    # Plot original vectors
    plot_vectors(ax_original, "(a) original vectors", vectors)

    # Define transformation matrix
    transformation_W = np.array([[2., 1.],[1.,2.]])

    # Compute transformed vectors
    transformed_vectors = {"W"+key: transformation_W @ vector for key, vector in vectors.items()}

    # Plot transformed vectors
    plot_vectors(ax_transformed, "(b) transformed vectors", transformed_vectors)

    # Print transformed vectors
    print("Basic transformation:")
    for key, transformed_vector in transformed_vectors.items():
        print(key + f" = {transformed_vector}")


    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()