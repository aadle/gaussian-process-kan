import argparse
from time import time
from collections import namedtuple
from functools import partial
import datetime

from data_setup import data_setup
from sklearn.model_selection import train_test_split

from jax import nn, numpy as jnp, random, config
from numpyro.contrib.einstein import RBFKernel, SteinVI, MixtureGuidePredictive
from numpyro import deterministic, plate, sample, set_platform, subsample
from numpyro.distributions import Gamma, Normal
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adagrad

DataState = namedtuple("data", ["xtr", "xte", "ytr", "yte"])


def load_data(x, y, args) -> DataState:
    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=args.test_size, random_state=1
    )

    return DataState(
        *map(partial(jnp.array, dtype=float), (xtr, xte, ytr, yte))
    )


def normalize(val, mean=None, std=None):
    """Normalize data to zero mean, unit variance"""
    if mean is None and std is None:
        # Only use training data to estimate mean and std.
        std = jnp.std(val, 0, keepdims=True)
        std = jnp.where(std == 0, 1.0, std)
        mean = jnp.mean(val, 0, keepdims=True)
    return (val - mean) / std, mean, std


def model(x, y=None, hidden_dim=32, sub_size=100):
    prec_nn = sample(
        "prec_nn", Gamma(1.0, 0.1)
    )  # hyper prior for precision of nn weights and biases

    n, m = x.shape

    # First hidden layer
    with plate("l1_hidden", hidden_dim, dim=-1):
        b1 = sample(
            "nn_b1",
            Normal(
                0.0,
                1.0 / jnp.sqrt(prec_nn),
            ),
        )
        assert b1.shape == (hidden_dim,)

        with plate("l1_feat", m, dim=-2):
            w1 = sample("nn_w1", Normal(0.0, 1.0 / jnp.sqrt(prec_nn)))
            assert w1.shape == (m, hidden_dim)

    # Second hidden layer
    with plate("l2_hidden", hidden_dim, dim=-1):
        b2 = sample(
            "nn_b2",
            Normal(
                0.0,
                1.0 / jnp.sqrt(prec_nn),
            ),
        )
        assert b2.shape == (hidden_dim,)

        with plate("l2_feat", hidden_dim, dim=-2):
            w2 = sample("nn_w2", Normal(0.0, 1.0 / jnp.sqrt(prec_nn)))
            assert w2.shape == (hidden_dim, hidden_dim)

    # Output layer
    with plate("l3_feat", hidden_dim, dim=-1):
        w3 = sample("nn_w3", Normal(0.0, 1.0 / jnp.sqrt(prec_nn)))
        assert w3.shape == (hidden_dim,)

    b3 = sample("nn_b3", Normal(0.0, 1.0 / jnp.sqrt(prec_nn)))

    prec_obs = sample("prec_obs", Gamma(1.0, 0.1))

    with plate("data", x.shape[0], subsample_size=sub_size, dim=-1):
        batch_x = subsample(x, event_dim=1)
        if y is not None:
            batch_y = subsample(y, event_dim=0)
        else:
            batch_y = y

        # Forward pass through both hidden layers
        h1 = nn.relu(batch_x @ w1 + b1)
        h2 = nn.relu(h1 @ w2 + b2)
        loc_y = deterministic("y_bnn", h2 @ w3 + b3)

        sample(
            "y",
            Normal(loc_y, 1.0 / jnp.sqrt(prec_obs)),
            obs=batch_y,
        )


def main(args):
    x1, x2, X, y = data_setup(args)
    data = load_data(X, y, args)
    x, xtr_mean, xtr_std = normalize(data.xtr)

    inf_key, pred_key, data_key = random.split(random.PRNGKey(args.rng_key), 3)
    rng_key, inf_key = random.split(inf_key)

    guide = AutoNormal(model)

    stein = SteinVI(
        model,
        guide,
        Adagrad(1.0),
        RBFKernel(),
        repulsion_temperature=args.repulsion,
        num_stein_particles=args.num_stein_particles,
        num_elbo_particles=args.num_elbo_particles,
    )
    start = time()

    result = stein.run(
        rng_key,
        args.max_iter,
        x,
        data.ytr,
        hidden_dim=args.hidden_dim,
        sub_size=args.subsample_size,
        progress_bar=args.progress_bar,
    )

    time_taken = time() - start

    pred = MixtureGuidePredictive(
        model,
        guide=stein.guide,
        params=stein.get_params(result.state),
        num_samples=100,
        guide_sites=stein.guide_sites,
    )

    xte, _, _ = normalize(
        data.xte, xtr_mean, xtr_std
    )  # Use train data statistics when accessing generalization.
    n = xte.shape[0]
    pred_y = pred(pred_key, xte, sub_size=n, hidden_dim=args.hidden_dim)["y"]
    rmse = jnp.sqrt(jnp.mean((pred_y.mean(0) - data.yte) ** 2))

    print(rf"Time taken: {datetime.timedelta(seconds=int(time_taken))}")
    print(rf"RMSE: {rmse:.2f}")


if __name__ == "__main__":
    config.update("jax_debug_nans", True)
    parser = argparse.ArgumentParser()

    # Data setup
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
    parser.add_argument("--test_size", nargs="?", default=0.2, type=float)

    # Numpyro arguments
    parser.add_argument("--subsample-size", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--repulsion", type=float, default=1.0)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--num-elbo-particles", type=int, default=50)
    parser.add_argument("--num-stein-particles", type=int, default=5)
    parser.add_argument("--progress-bar", type=bool, default=True)
    parser.add_argument("--rng-key", type=int, default=142)
    parser.add_argument("--device", default="cpu", choices=["gpu", "cpu"])
    parser.add_argument("--hidden-dim", default=32, type=int)

    args = parser.parse_args()

    set_platform(args.device)

    main(args)
