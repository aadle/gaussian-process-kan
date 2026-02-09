import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import optax
import tqdm

from jaxtyping import Float, Array, Int
from numpyro.contrib.module import nnx_module, random_nnx_module
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from flax_mlp import MLP
from flax import nnx
from sklearn.model_selection import train_test_split

from data_setup import data_setup, standardize_data 

# Bayesian Neural Network setup
nnx_module = MLP(2, 1, [16, 16], rngs=nnx.Rngs(0))
def bayesian_neural_net(
    x: Float[Array, "n_obs features"], y: Int[Array, "n_obs"] | None = None
) -> None:
    n_obs: int = x.shape[0]  # Number of observations

    # Prior distribution factory for network parameters.
    def prior(name, shape):
        if "bias" in name:
            return dist.Normal(loc=0, scale=1)
        return dist.SoftLaplace(loc=0, scale=1)

    # Create a NumPyro-wrapped version of our neural network
    # This automatically assigns priors to all parameters
    nn = random_nnx_module(
        "nn",  # Name prefix for all parameters
        nnx_module,  # Our Flax NNX module
        prior=prior,  # Prior distribution factory
    )
    loc_y = numpyro.deterministic("mean", nn(x).squeeze(-1))    
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs) # prior on obs?

    with numpyro.plate("data", n_obs):
        numpyro.sample("Y", dist.Normal(loc_y, sigma_obs), obs=y)
    
def training_loop(args, rng_key, x_train, y_train, x_val, y_val):
    rng_key, rng_subkey = jr.split(key=rng_key)

    # Model setup (own function?)
    optimizer = optax.adam(learning_rate=args.learning_rate)
    guide = AutoNormal(bayesian_neural_net)
    svi = SVI(
        model=bayesian_neural_net,
        guide=guide,
        optim=optimizer,
        loss=Trace_ELBO()
    )
    svi_state = svi.init(rng_subkey, x=x_train, y=y_train)

    train_step = jax.jit(
        lambda svi_state_: svi.update(svi_state_, x=x_train,
                                      y=y_train.squeeze())
    ) # returns (svi_state, loss)

    @jax.jit
    def get_val_loss(svi_state):
        _, rng_subkey = jr.split(svi_state.rng_key)
        params = svi.get_params(svi_state)  # Extract current parameter values

        # Compute loss without gradients or parameter updates
        return svi.loss.loss(
            rng_subkey,
            params,  # Current parameter values
            svi.model,  # Model function
            svi.guide,  # Guide function
            x=x_val,
            y=y_val.squeeze(),
        )
    
    batch = max(args.epochs// 20, 1)  # Batch size for progress updates
    patience = 100
    patience_counter = 0  # Counter for early stopping
    losses = {
        "training": [],
        "training_norm": [],
        "validation": [],
        "validation_norm": [],
    }
    with tqdm.trange(1, args.epochs + 1) as bar:

        for i in bar:
            svi_state, train_loss = train_step(svi_state)
            losses["training"].append(jax.device_get(train_loss))
            norm_train_loss = jax.device_get(train_loss)/y_train.shape[0]
            losses["training_norm"].append(norm_train_loss)

            val_loss = jax.jit(get_val_loss)(svi_state)
            losses["validation"].append(jax.device_get(val_loss))
            norm_val_loss = jax.device_get(val_loss)/y_val.shape[0]
            losses["validation_norm"].append(norm_val_loss)

            condition = norm_val_loss > norm_train_loss
            patience_counter = patience_counter + 1 if condition else 0

            if patience_counter >= patience:
                print(
                    f"Early stopping at step {i} (validation loss exceeding training loss)"
                )
                break
            if i % batch == 0:
                avg_train_loss = sum(losses["training"][i - batch :]) / batch
                avg_val_loss = sum(losses["validation"][i - batch :]) / batch

                bar.set_postfix_str(
                    f"train: {avg_train_loss:.4f}, val: {avg_val_loss:.4f}",
                    refresh=False,
                )
    
    # svi_result = SVIRunResult(
    #     params=svi.get_params(svi_state),
    #     state=svi_state,
    #     losses=losses["training"]
    # )

    return losses, svi_state


def main(args):
    rng_key = jr.key(args.seed)

    x1, x2, x, y = data_setup(args.function, args.num_samples)
    x_train_all, x_test, y_train_all, y_test = train_test_split(
        x, 
        y, 
        test_size=args.test_size,
        random_state=args.seed
    )

    # Try with non-standardized target, then with.
    x_train_sd, x_train_mean, x_train_std = standardize_data(x_train_all)
    # y_train_sd, y_train_mean, y_train_std = standardize_data(y_train_all)

    x_test_sd, _, _ = standardize_data(x_test, x_train_mean, x_train_std)
    # y_test_sd, _, _ = standardize_data(y_test, y_train_mean, y_train_std)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_sd, 
        # y_train_sd, 
        y_train_all, 
        test_size=args.test_size,
        random_state=args.seed
    )

    losses, svi_state = training_loop(
        args, 
        rng_key, 
        x_train_sd, 
        y_train, 
        x_val, 
        y_val
    )
    print(len(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian Neural Network training"
    )

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
    parser.add_argument("--num_samples", nargs="?", default=30, type=int)
    parser.add_argument("--test_size", nargs="?", default=0.3, type=float)

    parser.add_argument("--learning_rate", nargs="?", type=float, default=0.01)
    parser.add_argument("--epochs", nargs="?", type=int, default=3000)
    parser.add_argument("--seed", nargs="?", type=int, default=1)
    args = parser.parse_args()
    main(args)


