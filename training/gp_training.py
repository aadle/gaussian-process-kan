import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.use('Agg')
plt.ioff()

from sklearn.model_selection import train_test_split
from jaxtyping import install_import_hook
from data_setup import data_setup, standardize_data
from pathlib import Path
from training_utils import write_info_file

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
jax.config.update('jax_enable_x64', True)

def mse(y_true, y_pred):
    return jnp.mean((y_true.squeeze() - y_pred.squeeze()) ** 2)

def plot_training_history(history, output_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Negative MLL")
    ax.set_title("Training loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def init_gp(dataset, kernel):
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n)
    posterior = prior * likelihood

    return posterior

def get_gp_posterior(opt_posterior, x_in, train_data):
    latent_distribution = opt_posterior.predict(x_in, train_data=train_data)
    predictive_distribution = opt_posterior.likelihood(latent_distribution)
    predictive_mean = predictive_distribution.mean()
    predictive_std = predictive_distribution.stddev()
    return predictive_mean, predictive_std

def main(args):
    kernels = {
        "matern52": gpx.kernels.Matern52(),
        "matern32": gpx.kernels.Matern32(),
        "rbf": gpx.kernels.RBF(),
    }
    kernel = kernels.get(args.kernel)

    # === set up output directories ===
    cwd = Path()
    result_path = cwd/"results"
    result_path.mkdir(exist_ok=True)
    gp_path = result_path/"gp"
    gp_path.mkdir(exist_ok=True)
    output_path = gp_path/f"{args.kernel} {args.function}"
    output_path.mkdir(exist_ok=True)

    # === Set up data ===
    x1, x2, X, y = data_setup(args.function, args.n_samples)

    # Initial train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.key
    )

    # Standardize training data
    x_train_sd, x_train_mean, x_train_std = standardize_data(x_train)
    y_train_sd, y_train_mean, y_train_std = standardize_data(y_train)

    # Standardize training set using parameters from training set
    x_test_sd, _, _ = standardize_data(x_test, x_train_mean, x_train_std)
    # y_test_sd, _, _ = standardize_data(y_test, y_train_mean, y_train_std)

    train_dataset = gpx.Dataset(x_train_sd, y_train_sd)

    # === Set up GP ===
    posterior = init_gp(train_dataset, kernel)
    optimizer = optax.adam(args.learning_rate)

    # Training the GP
    key = jr.key(args.key)
    key, subkey = jr.split(key, 2)
    train_start = time.perf_counter()
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=train_dataset,
        optim=optimizer,
        num_iters=args.epochs,
        key=subkey,
        batch_size=args.batch_size
    )
    train_end = time.perf_counter()
    elapsed_training_time = train_end - train_start

    print("Negative Marginal log-likelihood: ",
          -gpx.objectives.conjugate_mll(opt_posterior, train_dataset)
    )

    plot_training_history(history, output_path/"loss.png")

    # === Predictions with optimized posterior ===

    # Predictions on test set
    test_start = time.perf_counter()
    test_pred_mean, test_pred_std = get_gp_posterior(
        opt_posterior=opt_posterior,
        x_in=x_test_sd,
        train_data=train_dataset
    )
    test_end = time.perf_counter()
    elapsed_test_time = test_end - test_start

    test_rescaled_pred_mean = test_pred_mean * y_train_std + y_train_mean
    test_rescaled_pred_std = test_pred_std * y_train_std
    test_mse = mse(y_test, test_rescaled_pred_mean)
    print("test MSE:", test_mse)

    # Predictions on entire dataset
    x_sd, _, _ = standardize_data(X, x_train_mean, x_train_std)
    pred_start = time.perf_counter()
    full_pred_mean, full_pred_std = get_gp_posterior(
        opt_posterior=opt_posterior,
        x_in=x_sd,
        train_data=train_dataset
    )
    pred_end = time.perf_counter()
    elapsed_pred_time = pred_end - pred_start

    rescaled_pred_mean = full_pred_mean * y_train_std + y_train_mean
    rescaled_pred_std = full_pred_std * y_train_std 
    full_mse = mse(y, rescaled_pred_mean)
    print("Full MSE:", full_mse)

    jnp.save(output_path/"mean_predictions.npy", rescaled_pred_mean)
    jnp.save(output_path/"sigma_predictions.npy", rescaled_pred_std)

    # write info into file...
    model_name = f"{args.function} with {args.kernel}"
    write_info_file(
        file_path=output_path/"info.json",
        model_name=model_name,
        x=X,
        x_train=x_train_sd,
        x_test=x_test_sd,
        args=args,
        elapsed_training_time=float(elapsed_training_time),
        elapsed_test_time=float(elapsed_test_time),
        elapsed_pred_time=float(elapsed_pred_time),
        test_mse=float(test_mse),
        full_mse=float(full_mse),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Process training")
    parser.add_argument("--learning_rate", nargs="?", default=1e-3, type=float)
    parser.add_argument("--n_samples", nargs="?", default=50, type=int)
    parser.add_argument("--epochs", nargs="?", default=500, type=int)
    parser.add_argument("--test_size", nargs="?", default=0.2, type=float)
    parser.add_argument("--batch_size", nargs="?", default=32, type=float)

    parser.add_argument(
        "--function",
        nargs="?",
        choices=[
            "himmelblau",
            "goldstein",
            "trig",
            "trollveggen",  
            "grandcanyon",
        ],
        default="himmelblau",
    )

    parser.add_argument(
        "--kernel",
        nargs="?",
        choices=[
            "rbf",
            "matern32",
            "matern52",
        ],
        default="rbf"
    )

    parser.add_argument("--key", nargs="?", default=123, type=int)

    args = parser.parse_args()

    main(args)
