# TODO:
# Development
# - [ ] In addition to the loss function, add an evaluation metric

# Features
# - [ ] Enable the possibility of importing/exporting model parameters
# - [ ]

import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
import tqdm
from gpkanmodel.model import GPKAN
from sklearn.model_selection import train_test_split
from data_setup import data_setup, standardize_data
from typing import Dict
from plotting import plot_results_normalized 

jax.config.update("jax_enable_x64", True)


def neg_loglikelihood(y_true, mean, covariance, jitter=1e-6):
    diag_elements = jnp.diag(covariance) + jitter
    covariance_inv = jnp.diag(1.0 / diag_elements)
    log_det = jnp.sum(jnp.log(diag_elements))
    y_true = y_true.flatten()

    return 0.5 * (
        y_true.shape[0] * jnp.log(2 * jnp.pi)
        + log_det
        + (y_true - mean).T @ covariance_inv @ (y_true - mean)
    )


def mse(y_true, y_pred):
    return jnp.mean((y_true.squeeze() - y_pred.squeeze()) ** 2)


def loss_fn(model_params, model, X_test, y_test, n_samples=10):
    latent_grids = model_params["latent_grids"]
    latent_supports = model_params["latent_supports"]
    kernel_parameters = model_params["kernel_parameters"]

    mean, covariance = model.sample_statistics(
        latent_grids,
        latent_supports,
        X_test,
        kernel_parameters,
        n_samples=n_samples,
    )

    return neg_loglikelihood(y_test, mean, covariance)

def training_step(
    model,
    model_params,
    opt_state,
    optimizer,
    batch_X,
    batch_y,
    val_and_grad_fn,
    mean_cov,
):
    loss, grads = val_and_grad_fn(
        model_params,
        batch_X,
        batch_y,
    )
    
    if loss < 0 or jnp.isnan(loss) or jnp.isinf(loss):
        mean, covariance = mean_cov(model_params, batch_X)
        batch_mse = mse(batch_y, mean)
        return model_params, opt_state, loss, batch_mse

    mean, covariance = mean_cov(model_params, batch_X)
    batch_mse = mse(batch_y, mean)
    
    updates, updated_opt_state = optimizer.update(grads, opt_state, model_params)
    updated_model_params = optax.apply_updates(model_params, updates)

    model.latent_grids = updated_model_params["latent_grids"]
    model.latent_supports = updated_model_params["latent_supports"]
    model.kernel_parameters = updated_model_params["kernel_parameters"]
       
    return updated_model_params, updated_opt_state, loss, batch_mse


def training_loop(args, model, x_train, y_train, key):
    val_and_grad_fn = jax.jit(
        jax.value_and_grad(
            lambda params, X, y: loss_fn(
                params,
                model,
                X,
                y,
                n_samples=10,
            ),
            argnums=0,
        )
    )

    mean_cov = jax.jit(
        lambda params, X:
            model.sample_statistics(
                params["latent_grids"],
                params["latent_supports"],
                X,
                params["kernel_parameters"],
                n_samples=10,
        )
    )

    model_params = {
        "latent_grids": model.latent_grids,
        "latent_supports": model.latent_supports,
        "kernel_parameters": model.kernel_parameters,
    }

    freeze_mask = {
        "latent_grids": args.freeze_x_latent,
        "latent_supports": False,
        "kernel_parameters": False, # Set to False to jointly optimize
    }
    lr = args.learning_rate
    optimizer = optax.transforms.selective_transform(
        optax.adam(learning_rate=lr), freeze_mask=freeze_mask
    )
    opt_state = optimizer.init(model_params)

    # Training history 
    loss_history = {
        "avg_nll": [],
        "avg_mse": [],
    }

    with tqdm.trange(1, args.epochs + 1) as t:
        for epoch in t:
            intra_epoch_loss = []
            intra_mse = []

            Switch optimizing latent support to kernel parameters
            midway_point = args.epochs/2
            if epoch == int(midway_point):
                mask = {
                    "latent_grids": args.freeze_x_latent,
                    "latent_supports": True,
                    "kernel_parameters": False,
                }
                scheduler_kernel = optax.linear_schedule(
                    args.learning_rate,
                    args.learning_rate * 1e-2,
                    midway_point
                )
                optimizer_chain = optax.chain(
                    optax.adam(scheduler_kernel),
                    optax.keep_params_nonnegative() # circumvent the problem of negative kernel parameters. Bijection is another option
                )
                optimizer = optax.transforms.selective_transform(
                    optimizer_chain, freeze_mask=mask
                )
                opt_state = optimizer.init(model_params)
                print("Optimizing kernel parameters")

            for i in range(0, x_train.shape[0], args.batch_size):
                key, subkey = jr.split(key)
                batch_X = x_train[i: i+args.batch_size, :]
                batch_y = y_train[i: i+args.batch_size, :]

                model_params, opt_state, loss, batch_mse = training_step(
                    model, 
                    model_params, 
                    opt_state, 
                    optimizer,
                    batch_X, 
                    batch_y,
                    val_and_grad_fn,
                    mean_cov
                )

                if not jnp.isnan(loss):
                    intra_epoch_loss.append(loss)
                    intra_mse.append(batch_mse)

                # Validation...

            if len(intra_epoch_loss) > 0:
                avg_nll = sum(intra_epoch_loss) / len(intra_epoch_loss)
                avg_mse = sum(intra_mse) / len(intra_mse)
                loss_history["avg_nll"].append(avg_nll)
                loss_history["avg_mse"].append(avg_mse)


            if epoch % 10 == 0 or epoch == 1:
                t.set_postfix_str(
                    f"NLL: {avg_nll:.4f}, MSE: {avg_mse:.4f}",
                refresh=False,
            )

            # if epoch % 10 == 0 or epoch == 0:
            #     print(
            #         f"Epoch {epoch}: Avg. NLL: {avg_nll:.4f}, Avg. MSE: {avg_mse:.4f}"
            #     )
    return loss_history

def prediction(args, model, model_params, X):
    batch_size = args.batch_size
    mu_batches = []
    sigma2_batches = []

    for i in tqdm.tqdm(range(0, X.shape[0], batch_size)):
        batch_X = X[i:i+batch_size]
        mu_batch, sigma2_batch = model.sample_statistics(
            model_params["latent_grids"], 
            model_params["latent_supports"], 
            batch_X, 
            model_params["kernel_parameters"], 
            20, 
            key=jr.key(i)
        )
        mu_batches.append(mu_batch)
        sigma2_batches.append(sigma2_batch)

    mu_full = jnp.concatenate(mu_batches)
    cov_full = jax.scipy.linalg.block_diag(*sigma2_batches)
    y_stddev = jnp.sqrt(jnp.diag(cov_full))

    return mu_full, y_stddev 
    

def save_parameters(model_params:Dict, path):
    path = ocp.test_utils.erase_and_create_empty(path)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path, model_params)

def restore_parameters(path):
    checkpointer = ocp.StandardCheckpointer()
    model_params = checkpointer.restore(path)
    return model_params


# TODO: 
# - [ ] Clean up the code.
# - [ ] Add optional to plot the uncertainty without standardizing it to the
# mean. See how that looks like.
# - [ ] Create utility file.
def main(args):
    model_name = ','.join(str(x) for x in args.model_size)

    # Data initialization
    x1, x2, X, y = data_setup(args.function, args.n_samples)

    match args.function:
        case "himmelblau":
            y = jnp.sqrt(y) # like in original implementation (himmelblau)
        case "goldstein":
            y = jnp.log(y)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.key
    )

    x_train_sd, x_train_mean, x_train_std = standardize_data(x_train)
    x_test_sd, _, _ = standardize_data(x_test, x_train_mean, x_train_std)

    # train-val split 
    # x_train, x_val, y_train, y_val = train_test_split(
    #     x_train_sd, y_train, test_size=args.test_size, random_state=args.key
    # )

    # Model initialization
    model = GPKAN(
        layers=args.model_size,
        n_grid_points=args.n_inducing,
        seed=args.key,
        grid_min=jnp.min(x_train_sd),
        grid_max=jnp.max(x_train_sd),
        init_paramters=[2.0, 2.0], 
    )

    # Training
    key = jr.PRNGKey(args.key)

    losses = training_loop(args, model, x_train_sd, y_train, key)

    # Loss and eval plots
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(losses["avg_nll"])
    ax[0].set_title("Negative log-likelihood")
    ax[1].plot(losses["avg_mse"])
    ax[1].set_title("Mean Squared Error")
    fig.suptitle("Loss and evaluation metrics during training")


    model_params = {
        "latent_grids": model.latent_grids,
        "latent_supports": model.latent_supports,
        "kernel_parameters": model.kernel_parameters,
    }

    # Test
    print("\n", "Predicting on test set...", "\n")
    mu_test, _ = prediction(args, model, model_params, x_test_sd)
    print("Test MSE:", mse(y_test.flatten(), mu_test.flatten()))
    

    # Do predictions on the entire dataset
    print("\n", "Predicting on full dataset...", "\n")
    x_std, _, _ = standardize_data(X, x_train_mean, x_train_std)
    mu, sigma = prediction(args, model, model_params, x_std)
    print("Full dataset MSE:", mse(y.flatten(), mu.flatten()))

    fig, ax = plot_results_normalized(
        x1,
        x2,
        y,
        mu,
        sigma,
    )
    plt.show()

    print(model_params["kernel_parameters"])

    # Save the figures

    # Save the parameters of the finally trained model...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gaussian Process Kolmogorov Arnold Network training"
    )

    # General arguments
    parser.add_argument("--key", nargs="?", default=123, type=int)

    # Model arguments
    parser.add_argument("--model_size", nargs="+", default=[2, 5, 1], type=int)  
    parser.add_argument("--n_inducing", nargs="?", default=10, type=int)
    parser.add_argument(
        "--freeze_x_latent",
        nargs="?",
        choices=[True, False],
        default=True,
        type=bool,
    )

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
    parser.add_argument("--n_samples", nargs="?", default=50, type=int)

    # Training loop arguments
    parser.add_argument("--epochs", nargs="?", default=200, type=int)
    parser.add_argument("--learning_rate", nargs="?", default=1e-2, type=float)
    parser.add_argument("--batch_size", nargs="?", default=32, type=int)
    parser.add_argument("--test_size", nargs="?", default=0.2, type=float)

    args = parser.parse_args()

    main(args)
