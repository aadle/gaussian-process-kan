# TODO:
# - [ ] Enable the possibility of importing/exporting model parameters
# - [ ] Add a validation step after training step
# - [ ] Proper transformation of data for 'trollveggen' and 'grand canyon'

import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
import tqdm
import time
import matplotlib

matplotlib.use('Agg')

from flax import nnx
from gpkanmodel.model import GPKAN
from sklearn.model_selection import train_test_split
from data_setup import data_setup, standardize_data
from typing import Dict
from plotting import plot_results_normalized 
from gpjax.parameters import transform, DEFAULT_BIJECTION
from training_utils import write_info_file
from pathlib import Path

jax.config.update("jax_enable_x64", True)
plt.ioff()


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


def transform_params(params, inverse:bool=True):
    return jax.tree_util.tree_map(
        lambda s: transform(s, DEFAULT_BIJECTION, inverse=inverse),
        params,
        is_leaf=lambda x: isinstance(x, nnx.State)
    )

def loss_fn(model_params, model, X_test, y_test, n_samples=10):
    latent_grids = model_params["latent_grids"]
    latent_supports = model_params["latent_supports"]
    kernel_parameters = transform_params(
        model_params["kernel_parameters"], 
        inverse=False
    )

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
    key
):
    loss, grads = val_and_grad_fn(
        model_params,
        batch_X,
        batch_y,
    )
    
    if jnp.isnan(loss) or jnp.isinf(loss):
        return model_params, opt_state, loss, jnp.nan

    mean, covariance = mean_cov(model_params, batch_X)
    batch_mse = mse(batch_y, mean)
    
    updates, updated_opt_state = optimizer.update(grads, opt_state, model_params)
    updated_model_params = optax.apply_updates(model_params, updates)

    model.latent_grids = updated_model_params["latent_grids"]
    model.latent_supports = updated_model_params["latent_supports"]
    model.kernel_parameters = updated_model_params["kernel_parameters"]
       
    return updated_model_params, updated_opt_state, loss, batch_mse

def get_mean_cov(model, params, X):
    kernel_parameters = transform_params(params["kernel_parameters"], inverse=False)
    return model.sample_statistics(
        params["latent_grids"],
        params["latent_supports"],
        X,
        kernel_parameters,
        n_samples=10,
    )


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
        lambda params_, X_: get_mean_cov(model, params_, X_)
    )

    model_params = {
        "latent_grids": model.latent_grids,
        "latent_supports": model.latent_supports,
        "kernel_parameters": model.kernel_parameters,
    }
    model_params["kernel_parameters"] = transform_params(
        model_params["kernel_parameters"], 
        inverse=True
    )

    total_steps = args.epochs*(jnp.ceil(x_train.shape[0]/args.batch_size)) + args.epochs
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=args.learning_rate/100,
        peak_value=args.learning_rate,
        warmup_steps=int(0.3 * total_steps),
        decay_steps=total_steps,
        end_value=args.learning_rate/10,
    )

    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(model_params)

    # Training history 
    loss_history = {
        "train_nll": [],
        "train_mse": [],
    }

    with tqdm.trange(1, args.epochs+1) as t:
        for epoch in t:
            intra_nll = []
            intra_mse = []

            for i in range(0, x_train.shape[0], args.batch_size):
                key, subkey = jr.split(key)
                batch_x = x_train[i: i+args.batch_size, :]
                batch_y = y_train[i: i+args.batch_size, :]

                model_params, opt_state, loss, batch_mse = training_step(
                    model, 
                    model_params, 
                    opt_state, 
                    optimizer,
                    batch_x, 
                    batch_y,
                    val_and_grad_fn,
                    mean_cov,
                    subkey
                )
                intra_nll.append(loss)
                intra_mse.append(batch_mse)

            train_nll = sum(intra_nll) / args.batch_size
            train_mse = sum(intra_mse) / args.batch_size
            loss_history["train_nll"].append(train_nll)
            loss_history["train_mse"].append(train_mse)

            if epoch % 10 == 0 or epoch == 1:
                t.set_postfix_str(
                    f"NLL: {train_nll:.4f}, MSE: {train_mse:.4f}",
                refresh=False,
            )
    return loss_history

def prediction(args, model, model_params, X):
    batch_size = args.batch_size
    mu_batches = []
    sigma2_batches = []

    for i in tqdm.tqdm(range(0, X.shape[0], batch_size)):
        # key, subkey = jr.split(key)
        batch_X = X[i:i+batch_size]
        mu_batch, sigma2_batch = model.sample_statistics(
            model_params["latent_grids"], 
            model_params["latent_supports"], 
            batch_X, 
            model_params["kernel_parameters"], 
            20, 
            key=jr.key(i)
        ) # TODO: check why using a split key does not work...
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

def main(args):
    working_dir = Path()
    model_name = '-'.join(str(x) for x in args.model_size)
    print("Model size:", model_name)
    filename = model_name + " " + args.function

    result_path = working_dir/"results"
    result_path.mkdir(exist_ok=True)

    dir_path = result_path/filename
    dir_path.mkdir(exist_ok=True)

    # Data initialization
    x1, x2, X, y = data_setup(args.function, args.n_samples)

    # Initial train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.key
    )

    # Standardize training data
    x_train_sd, x_train_mean, x_train_std = standardize_data(x_train)
    y_train_sd, y_train_mean, y_train_std = standardize_data(y_train)

    # Apply standardization to test data
    x_test_sd, _, _ = standardize_data(x_test, x_train_mean, x_train_std)

    # Train-validation split 
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

    # Training loop
    key = jr.PRNGKey(args.key)
    train_start = time.perf_counter()
    losses = training_loop(args, model, x_train_sd, y_train_sd, key)
    train_end = time.perf_counter()
    elapsed_training_time = train_end - train_start

    # Loss and evaluation plots of training
    fig_loss, ax_loss = plt.subplots(nrows=2)
    ax_loss[0].plot(losses["train_nll"])
    ax_loss[0].set_title("Negative log-likelihood")
    ax_loss[1].plot(losses["train_mse"])
    ax_loss[1].set_title("Mean Squared Error")
    fig_loss.suptitle("Loss and evaluation metrics during training")
    plt.tight_layout()

    loss_eval_filename = filename+"_loss_eval.png"
    fig_loss.savefig(dir_path/loss_eval_filename, dpi=500)
    plt.close()

    # Retrieve kernel parameters from trained model and transform back to original
    # parameter space
    model_params = {
        "latent_grids": model.latent_grids,
        "latent_supports": model.latent_supports,
        "kernel_parameters": model.kernel_parameters,
    }
    model_params["kernel_parameters"] = transform_params(
        model_params["kernel_parameters"], 
        inverse=False
    )

    # Prediction on the test set
    print("\n", "Predicting on test set...", "\n")
    test_start = time.perf_counter()
    y_hat_test, _ = prediction(args, model, model_params, x_test_sd)
    test_end = time.perf_counter()
    y_hat_test_rescaled = y_hat_test * y_train_std + y_train_mean # rescaling to original scale
    test_mse = mse(y_hat_test_rescaled.flatten(), y_test.flatten())
    elapsed_test_time = test_end - test_start
    print(f"Test MSE: {test_mse:.6f}")


    # Prediction on the full dataset
    print("\n", "Predicting on full dataset...", "\n")
    x_std, _, _ = standardize_data(X, x_train_mean, x_train_std) # standardize data w.r.t. training set parameters
    pred_start = time.perf_counter()
    mu_hat, sigma = prediction(args, model, model_params, x_std)
    pred_end = time.perf_counter()
    sigma_hat_rescaled = sigma * y_train_std # rescaling to original scale
    mu_hat_rescaled = mu_hat * y_train_std + y_train_mean # rescaling to original scale
    full_mse = mse(y.flatten(), mu_hat_rescaled.flatten())
    elapsed_pred_time = pred_end - pred_start
    print(f"Full dataset MSE: {full_mse:.6f}")

    # Plotting full results
    model_size = "-".join([str(x) for x in args.model_size])
    fig_pred, ax_pred = plot_results_normalized(
        x1,
        x2,
        y,
        mu_hat_rescaled,
        sigma_hat_rescaled,
        figsize=(20, 5),
        title=f"{model_size} GPKAN, MSE: {full_mse:.4f}",
        clip_outliers=True
    )
    results_filename = filename+"_results.png"
    fig_pred.savefig(dir_path/results_filename, dpi=500)
    plt.close()
    # plt.show()

    # Save the parameters of the finally trained model...
    # TODO: Does not work properly?
    # params_path = dir_path/"model_params"
    # save_parameters(model_params=model_params, path=params_path.absolute())

    # Save the predictions...
    mean_pred_filename = f"{model_name} mean_predictions.npy"
    sigma_pred_filename = f"{model_name} sigma_predictions.npy"
    jnp.save(dir_path/mean_pred_filename, mu_hat_rescaled)
    jnp.save(dir_path/sigma_pred_filename, sigma_hat_rescaled)

    # Data describing script run
    write_info_file(
        file_path=dir_path / "info.json",
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

