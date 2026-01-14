import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np


def plot_2d_predictions(
    x1,
    x2,
    y,
    mu,
    residuals,
    y_stddev,
    title="Results",
    figsize=(8, 8),
    cmap1="viridis",
    cmap2="jet",
    dpi=100,
):
    """
    Create a 2x2 contour plot comparing true data, predictions, residuals, and uncertainty.

    Parameters
    ----------
    x1 : array-like
        1D grid coordinates for first dimension
    x2 : array-like
        1D grid coordinates for second dimension
    y : array-like
        True underlying data (flattened or can be reshaped)
    mu : array-like
        Predicted mean values (same shape as y)
    residuals : array-like
        Residuals: mu - y (same shape as y)
    y_stddev : array-like
        Standard deviation / uncertainty estimates (same shape as y)
    title : str, optional
        Title for the overall figure (default: "Results")
    figsize : tuple, optional
        Figure size (default: (8, 8))
    cmap1 : str, optional
        Colormap for top two plots (default: "viridis")
    cmap2 : str, optional
        Colormap for bottom two plots (default: "jet")
    dpi : int, optional
        DPI for display (default: 100)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axs : array of matplotlib.axes.Axes
        The axes objects (flattened 2x2 grid)
    """

    fig, axs = plt.subplots(
        2, 2, figsize=figsize, constrained_layout=True, dpi=dpi
    )
    axs = axs.flatten()

    # Ensure data is flattened for reshaping operations
    y_flat = np.asarray(y).flatten()
    mu_flat = np.asarray(mu).flatten()
    residuals_flat = np.asarray(residuals).flatten()
    stddev_flat = np.asarray(y_stddev).flatten()

    # ========== TOP ROW: Data and Predictions ==========

    # Determine shared min/max for top plots
    vmin = min(y_flat.min(), mu_flat.min())
    vmax = max(y_flat.max(), mu_flat.max())
    norm1 = plt.Normalize(vmin, vmax)

    # Plot 1: Actual data
    axs[0].contourf(
        x1,
        x2,
        y_flat.reshape(x2.shape[0], x1.shape[0]),
        levels=50,
        cmap=cmap1,
        vmin=vmin,
        vmax=vmax,
    )
    axs[0].set_title("Underlying data")
    axs[0].set_xlabel("$x_1$")
    axs[0].set_ylabel("$x_2$")

    # Plot 2: Predicted mean
    axs[1].contourf(
        x1,
        x2,
        mu_flat.reshape(x2.shape[0], x1.shape[0]),
        cmap=cmap1,
        levels=50,
        vmin=vmin,
        vmax=vmax,
    )
    axs[1].set_title("Approximated Mean Function")
    axs[1].set_xlabel("$x_1$")
    axs[1].set_ylabel("$x_2$")

    # Colorbar for top row
    sm1 = ScalarMappable(cmap=cmap1, norm=norm1)
    sm1.set_array([])
    cbar_row1 = fig.colorbar(
        sm1, ax=[axs[0], axs[1]], location="right", shrink=0.98
    )
    cbar_row1.set_label("Function value")

    n_ticks = 9
    ticks1 = np.linspace(vmin, vmax, n_ticks)
    cbar_row1.set_ticks(ticks1)
    cbar_row1.set_ticklabels([f"{tick:.2f}" for tick in ticks1])

    # ========== BOTTOM ROW: Residuals and Uncertainty ==========

    # Calculate normalized metrics
    epsilon = 1e-10
    normalized_residuals = (
        100 * np.abs(residuals_flat) / (np.abs(mu_flat) + epsilon)
    )
    normalized_uncertainty = 100 * (stddev_flat / (np.abs(mu_flat) + epsilon))

    vmin_2 = min(normalized_residuals.min(), normalized_uncertainty.min())
    vmax_2 = max(normalized_residuals.max(), normalized_uncertainty.max())
    cbar_limit = min(vmax_2, 100)
    levels_2 = np.linspace(vmin_2, cbar_limit, 20)

    # Plot 3: Normalized residuals
    axs[2].contourf(
        x1,
        x2,
        normalized_residuals.reshape(x2.shape[0], x1.shape[0]),
        levels=levels_2,
        cmap=cmap2,
        vmin=vmin_2,
        vmax=cbar_limit,
        extend="max",
    )
    axs[2].set_title("Normalized Residuals")
    axs[2].set_xlabel("$x_1$")
    axs[2].set_ylabel("$x_2$")

    # Plot 4: Normalized uncertainty
    axs[3].contourf(
        x1,
        x2,
        normalized_uncertainty.reshape(x2.shape[0], x1.shape[0]),
        levels=levels_2,
        cmap=cmap2,
        vmin=vmin_2,
        vmax=cbar_limit,
        extend="max",
    )
    axs[3].set_title("Normalized Uncertainty")
    axs[3].set_xlabel("$x_1$")
    axs[3].set_ylabel("$x_2$")

    # Colorbar for bottom row
    norm2 = plt.Normalize(vmin_2, cbar_limit)
    sm2 = ScalarMappable(cmap=cmap2, norm=norm2)
    sm2.set_array([])
    cbar_row2 = fig.colorbar(
        sm2, ax=[axs[2], axs[3]], location="right", shrink=0.98, extend="max"
    )
    cbar_row2.set_label("Relative Error (%)")

    step_size = 10 if cbar_limit > 50 else 2.5

    if cbar_limit > 50:
        step_size = 10
    elif cbar_limit < 50:
        step_size = 5
    elif cbar_limit < 20:
        step_size = 2.5

    ticks2 = np.arange(0, cbar_limit + 1, step_size)
    cbar_row2.set_ticks(ticks2)
    cbar_row2.set_ticklabels([f"{tick}" for tick in ticks2])

    # Overall title
    fig.suptitle(title, fontsize=30, fontweight="bold")

    return fig, axs


# Example usage:
if __name__ == "__main__":
    # Create sample 2D data
    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-2, 2, 50)
    X1, X2 = np.meshgrid(x1, x2)

    # Goldstein-Price function
    part1 = 1 + (X1 + X2 + 1) ** 2 * (
        19 - 14 * X1 + 3 * X1**2 - 14 * X2 + 6 * X1 * X2 + 3 * X2**2
    )
    part2 = 30 + (2 * X1 - 3 * X2) ** 2 * (
        18 - 32 * X1 + 12 * X1**2 + 48 * X2 - 36 * X1 * X2 + 27 * X2**2
    )
    y = np.log(part1 * part2)

    # Create synthetic predictions and uncertainty
    mu = y + np.random.randn(*y.shape) * 0.1
    residuals = mu - y
    y_stddev = np.abs(np.random.randn(*y.shape) * 0.15)

    # Create the plot
    fig, axs = plot_2d_predictions(
        x1, x2, y, mu, residuals, y_stddev, title="Goldstein-Price Function"
    )
    plt.show()
