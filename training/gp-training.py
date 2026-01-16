# TODO:
# - [ ] Train the GP with log-likelihood, as done previously in the thesis
# codebase.

import jax
import jax.numpy as jnp
import argparse
import matplotlib.pyplot as plt
from data_setup import data_setup
from plotting import plot_2d_predictions
from jaxtyping import install_import_hook
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
jax.config.update('jax_enable_x64', True)

def init_gp(X, y):
    D = gpx.Dataset(X, y)
    kernel = gpx.kernels.RBF()
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    posterior = prior * likelihood

    return posterior

def main(args):
    x1, x2, X, y = data_setup(args)
    y = jnp.log(y)
    D = gpx.Dataset(X=X, y=y)
    posterior = init_gp(X, y)

    # training with marginal log-likelihood
    opt_posterior, history = gpx.fit_scipy(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=D,
    )

    print(-gpx.objectives.conjugate_mll(opt_posterior, D))

    # training with log-likelihood
    # When optimizing, sample approach or use GP posterior distribution
    # parameters as input for the loss function?

    # Prediction with optimized GP posterior
    latent_dist = opt_posterior.predict(X, train_data=D)
    predictive_dist = opt_posterior.likelihood(latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()


    res = y.flatten() - predictive_mean
    fig = plot_2d_predictions(x1, x2, y=y, mu=predictive_mean, residuals=res, y_stddev=predictive_std)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=" Gaussian Process training"
    )
    # Data setup arguments
    parser.add_argument(
        "--function",
        nargs="?",
        choices=[
            "himmelblau",
            "goldstein",
            "trig",
            "trollveggen"  # not implemented yet
            "grandcanyon",  # not implemented yet
        ],
        default="himmelblau",
    )
    parser.add_argument("--n_samples", nargs="?", default=20, type=int)
    args = parser.parse_args()
    main(args)

