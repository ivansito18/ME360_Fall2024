import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})

def plot_transformation(matrix_W: np.ndarray, vectors: dict[str, np.ndarray]) -> None:
    fig, (ax_original, ax_transformed) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original vectors

    ax_original.set_title("(a) original vectors")
    ax_original.set_xlim(-5, 5)
    ax_original.set_ylim(-5, 5)
    ax_original.axhline(y=0, color='k', linestyle='--')
    ax_original.axvline(x=0, color='k', linestyle='--')
    ax_original.grid(True, linestyle=':', alpha=0.7)

    for i, (key, vector) in enumerate(vectors.items()):
        ax_original.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=f"C{i}", label="$"+key+"$")


    # Plot transformed vectors

    ax_transformed.set_title("(b) after transformation")
    ax_transformed.set_xlim(-5, 5)
    ax_transformed.set_ylim(-5, 5)

    projection_color = 'gray'
    s = np.linspace(-5, 5, 100)
    ax_transformed.plot(matrix_W[0] * s, matrix_W[1] * s, color=projection_color, linestyle="--", alpha=0.5)
    ax_transformed.quiver(0, 0, matrix_W[0], matrix_W[1], angles='xy', scale_units='xy', scale=1, color=projection_color, label="$W$")

    for i, (key, vector) in enumerate(vectors.items()):
        ax_transformed.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=f"C{i}", label="$"+key+"$", alpha=0.5)
    for i, (key, vector) in enumerate(vectors.items()):
        transformed_v = matrix_W @ vector
        unit_vector = matrix_W / np.linalg.norm(matrix_W)
        ax_transformed.quiver(0, 0, transformed_v*unit_vector[0], transformed_v*unit_vector[1], angles='xy', scale_units='xy', scale=1, color=f"C{i}", label="$W"+key+"$")
    
    for ax in (ax_original, ax_transformed):
        ax.set_aspect('equal')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def main() -> None:
    # Define vectors dictionary
    vectors = {
        'v_a': np.array([-2, 1]),
        'v_b': np.array([1, 1]),
        'v_c': np.array([0, -2]),
    }
    
    # Basic transformation
    transformation_W = np.array([1, 2])
    print("Basic transformation:")
    for key, vector in vectors.items():
        result = transformation_W @ vector
        print("W @ " + key + f" = {result}")
    
    # Plot transformation
    plot_transformation(transformation_W, vectors)


if __name__ == "__main__":
    main()