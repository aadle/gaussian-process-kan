from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from data_setup import data_setup
from matplotlib.cm import ScalarMappable
from matplotlib.colors import SymLogNorm


def plot_results_normalized(
    x1,
    x2,
    y,
    mu_hat,
    sigma_hat,
    clip_outliers=False,
    log_scale=False,
    title="Results",
    subplot_size=5,
    cmap1="viridis",
    cmap2="jet",
    dpi=100,
):
    figsize = (subplot_size * 2 + 1.5, subplot_size * 2)
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True, dpi=dpi)
    axs = axs.flatten()
    epsilon = 1e-6

    y_flat = np.asarray(y).flatten()
    mu_hat_flat = np.asarray(mu_hat).flatten()
    stddev_flat = np.asarray(sigma_hat).flatten()

    vmin = min(y_flat.min(), mu_hat_flat.min())
    vmax = max(y_flat.max(), mu_hat_flat.max())

    if log_scale:
        vmin_safe = max(vmin, epsilon)
        norm1 = SymLogNorm(linthresh=0.1, vmin=vmin_safe, vmax=vmax)
    else:
        norm1 = plt.Normalize(vmin, vmax)

    # Top row of plots
    for i, (data, lbl) in enumerate(
        [(y_flat, "Underlying Data"), (mu_hat_flat, "Predicted Mean")]
    ):
        axs[i].contourf(
            x1,
            x2,
            data.reshape(x2.shape[0], x1.shape[0]),
            levels=50,
            cmap=cmap1,
            norm=norm1,
        )
        axs[i].set_title(lbl)
        axs[i].set_xlabel("$x_1$")
        axs[i].set_ylabel("$x_2$")

    sm1 = ScalarMappable(cmap=cmap1, norm=norm1)
    fig.colorbar(sm1, ax=[axs[0], axs[1]], location="right", shrink=0.8, label="Value")

    # Bottom row of plots
    def nice_step(data_range, target_ticks=8):
        """Pick a round step size that gives ~target_ticks ticks."""
        raw_step = data_range / target_ticks
        magnitude = 10 ** np.floor(np.log10(raw_step))
        for multiplier in [1, 2, 5, 10]:
            step = multiplier * magnitude
            if data_range / step <= target_ticks:
                return step
        return multiplier * magnitude

    def plot_metric(ax, data, title_str, clip_outliers=False):
        d_max = np.percentile(data, 95) if clip_outliers else data.max()
        d_min = data.min()

        data_range = d_max - d_min
        step = nice_step(data_range)

        if data_range >= 20:
            step = max(10.0, round(step / 10) * 10)

        vmin_clean = np.floor(d_min / step) * step
        vmax_clean = np.ceil(d_max / step) * step
        levels = np.linspace(vmin_clean, vmax_clean, 50)

        cf = ax.contourf(
            x1,
            x2,
            data.reshape(x2.shape[0], x1.shape[0]),
            levels=levels,
            cmap=cmap2,
            vmin=vmin_clean,
            vmax=vmax_clean,
            extend="max",
        )
        ax.set_title(title_str)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        cbar = fig.colorbar(cf, ax=ax, location="right", shrink=0.8)
        cbar.set_label("% Relative to Mean")

        tick_locs = np.arange(vmin_clean, vmax_clean + step, step)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([f"{t:.4g}" for t in tick_locs])

    scaled_res = np.abs((y_flat - mu_hat_flat) / (mu_hat_flat + epsilon)) * 100
    plot_metric(
        axs[2], scaled_res, "Mean Scaled Residuals", clip_outliers=clip_outliers
    )

    scaled_std = np.abs(stddev_flat / (mu_hat_flat + epsilon)) * 100
    plot_metric(
        axs[3],
        scaled_std,
        "Mean Scaled Standard Deviation",
        clip_outliers=clip_outliers,
    )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    return fig, axs


def main():
    results_dir = Path("results")

    for model_type_dir in [results_dir / "gp", results_dir / "gpkan"]:
        if not model_type_dir.exists():
            continue

        for run_dir in sorted(model_type_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            dir_name = run_dir.name  # e.g. "rbf himmelblau" or "2-5-5-1 himmelblau"
            _, fn_name = dir_name.split(" ", 1)

            mu_path = run_dir / "mean_predictions.npy"
            sigma_path = run_dir / "sigma_predictions.npy"

            if not mu_path.exists() or not sigma_path.exists():
                print(f"Skipping {run_dir.name}: prediction files not found")
                continue

            n_samples = 50
            mu_hat = np.load(mu_path)
            sigma_hat = np.load(sigma_path)
            x1, x2, _, y = data_setup(fn_name, n_samples)

            fig, _ = plot_results_normalized(
                x1,
                x2,
                y,
                mu_hat,
                sigma_hat,
                clip_outliers=True,
                log_scale=False,
                title=f"{dir_name} ({model_type_dir.name})",
            )

            out_path = run_dir / f"replot {dir_name}.png"
            fig.savefig(out_path, dpi=500)
            plt.close(fig)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
