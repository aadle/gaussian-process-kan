# TODO:
# - [ ] Train the GP with log-likelihood, as done previously in the thesis
# codebase.

import jax
import jax.numpy as jnp
import jax.random as jr
import argparse
import matplotlib.pyplot as plt
import optax
from data_setup import data_setup
from plotting import plot_2d_predictions, plot_2d_results
from jaxtyping import install_import_hook
from sklearn.model_selection import train_test_split
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

jax.config.update('jax_enable_x64', True)

def mse(y_true, y_pred):
    return jnp.mean((y_true.squeeze() - y_pred.squeeze()) ** 2)

def init_gp(dataset, args):
    match args.kernel:
        case "matern52":
            kernel = gpx.kernels.Matern52()
        case "matern32":
            kernel = gpx.kernels.Matern32()
        case _:
            kernel = gpx.kernels.RBF()

    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n)
    posterior = prior * likelihood

    return posterior

def main(args):
    key = jr.key(args.key)

    x1, x2, X, y = data_setup(args)

    match args.function:
        case "himmelblau":
            y = jnp.sqrt(y) # like from the thesis code
        case "goldstein":
            y = jnp.log(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.key
    )

    # Standardize training data if the ranges are quite large.
    if args.standardize:
        X_train_mean = jnp.mean(X_train, axis=0)
        X_train_stddev = jnp.std(X_train, axis=0)
        X_train_std = (X_train - X_train_mean) / X_train_stddev
        y_train_mean = jnp.mean(y_train)
        y_train_stddev = jnp.mean(y_train)
        y_train_std = (y_train - y_train_mean) / y_train_stddev

        D = gpx.Dataset(X=X_train_std, y=y_train_std)
    else:
        D = gpx.Dataset(X=X_train, y=y_train)

    posterior = init_gp(D, args)

    # training with marginal log-likelihood.
    # https://docs.jaxgaussianprocesses.com/_examples/regression/#parameter-state
    # opt_posterior, history = gpx.fit_scipy(
    #     model=posterior,
    #     objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    #     train_data=D,
    # )
    
    # Alternatively with more control
    optimizer = optax.adam(args.learning_rate)
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=D,
        optim=optimizer,
        num_iters=500,
        key=key,
        batch_size=args.batch_size
    )

    print("Negative Marginal log-likelihood: ", -gpx.objectives.conjugate_mll(opt_posterior, D))

    # training with log-likelihood
    # When optimizing, sample approach or use GP posterior distribution
    # parameters as input for the loss function?

    # Prediction with optimized GP posterior

    if args.standardize:
        X = X * X_train_stddev + X_train_mean

    latent_dist = opt_posterior.predict(X, train_data=D)
    predictive_dist = opt_posterior.likelihood(latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()

    if args.standardize:
        predictive_mean = predictive_mean * y_train_stddev + y_train_mean # scale back
        predictive_std = predictive_std * y_train_stddev + y_train_mean # scale back

    res = y.flatten() - predictive_mean
    fig, axs = plot_2d_predictions(
        x1, 
        x2, 
        y=y, 
        mu=predictive_mean, 
        residuals=res, 
        y_stddev=predictive_std
    )
    fig, axs = plot_2d_results(
        x1, 
        x2, 
        y, 
        predictive_mean, 
        predictive_std
    )
    plt.show()
    
    print("MSE:", mse(y, predictive_mean))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=" Gaussian Process training"
    )

    # General arguments
    parser.add_argument("--key", nargs="?", default=123, type=int)

    # Data setup arguments
    parser.add_argument(
        "--function",
        nargs="?",
        choices=[
            "himmelblau",
            "goldstein",
            "trig",
            "trollveggen",
            "grandcanyon",  # not implemented yet
        ],
        default="himmelblau",
    )
    parser.add_argument("--n_samples", nargs="?", default=20, type=int)
    parser.add_argument("--standardize", nargs="?", default=False, type=bool)

    # Model arguments
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

    # Training arguments
    parser.add_argument("--learning_rate", nargs="?", default=1e-3, type=float)
    parser.add_argument("--test_size", nargs="?", default=0.2, type=float)
    parser.add_argument("--batch_size", nargs="?", default=128, type=int)

    args = parser.parse_args()
    main(args)

