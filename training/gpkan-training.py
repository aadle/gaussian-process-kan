# TODO:
    # Development
    # - [ ] In addition to the loss function, add an evaluation metric

    # Features
    # - [ ] Enable the possibility of importing/exporting model parameters
    # - [ ]

import argparse
import jax
import jax.numpy as jnp
from gpkanmodel.model import GPKAN
from sklearn.model_selection import train_test_split
from data_setup import data_setup

jax.config.update("jax_enable_x64", True)

def neg_loglikelihood(y_true, mean, covariance):
    diag_elements = jnp.diag(covariance)
    covariance_inv = jnp.diag(1.0 / diag_elements)
    log_det = jnp.sum(jnp.log(diag_elements))
    y_true = y_true.flatten()

    return 0.5 * (
        y_true.shape[0] * jnp.log(2 * jnp.pi)
        + log_det
        + (y_true - mean).T @ covariance_inv @ (y_true - mean)
    )

def loss_fn(model_params, model, X_test, y_test, n_samples=10):
    latent_grids, latent_supports, kernel_params = model_params
    mean, covariance = model.sample_statistics(
        latent_grids, 
        latent_supports, 
        X_test, 
        kernel_params,
        n_samples=n_samples
    )
    return neg_loglikelihood(y_test, mean, covariance)

def training_loop(args, model, X_train, y_train):
    val_grad_loss = jax.jit(
        jax.value_and_grad(
            lambda Xs_latent, ys_latent, kernel_params, X_test, y_test:
            neg_loglikelihood(
                y_test,
                *model.sample_statistics(
                    Xs_latent, 
                    ys_latent, 
                    X_test, 
                    kernel_params, 
                    n_samples=10
                )
            ),
            argnums=(0, 1, 2)
        )
    )

    model_params = (
        model.latent_grids,
        model.latent_supports,
        model.kernel_parameters
    )

    loss_history = []
    for epoch in range(args.epochs):
        intra_epoch_loss = []

        for i in range(0, X_train.shape[0], args.batch_size):
            batch_X = X_train[i:i+args.batch_size, :]
            batch_y = y_train[i:i+args.batch_size, :]

            loss, (grad_grids, grad_supports, grad_params) = val_grad_loss(
                model.latent_grids, 
                model.latent_supports,
                model.kernel_parameters,
                batch_X, batch_y
                )
            intra_epoch_loss.append(loss)

            model.latent_supports = jax.tree.map(
                lambda latent_supports, grad_supports_: 
                latent_supports - grad_supports_ * args.learning_rate,
                model.latent_supports,
                grad_supports
            )

        avg_loss = sum(intra_epoch_loss) / len(intra_epoch_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg. loss: {avg_loss:.6f}, LR: {args.learning_rate:.6f}")
        loss_history.append(avg_loss)

def main(args, model: GPKAN):
    X, y = data_setup(args)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size, 
        random_state=321
    )
    y_train_std = (y_train - jnp.mean(y_train) ) / jnp.std(y_train)
    training_loop(args, model, X_train, y_train_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gaussian Process Kolmogorov Arnold Network training"
    )

    # Model arguments
    parser.add_argument("--model_size", nargs="?", default=[2, 5, 1], type=list)
    # Does not work properly

    # Data setup arguments
    parser.add_argument(
        "--function", 
        nargs="?", 
        choices=[
        "himmelblau", 
        "goldstein", 
        "trig", 
        "trollveggen" # not implemented yet
        "grandcanyon" # not implemented yet
        ],
        default="himmelblau"
    )
    parser.add_argument("--num_samples", nargs="?", default=20, type=int)

    # Training loop arguments
    parser.add_argument("--epochs", nargs="?", default=500, type=int)
    parser.add_argument("--learning_rate", nargs="?", default=0.01, type=float)
    parser.add_argument("--batch_size", nargs="?", default=32, type=float)
    parser.add_argument("--test_size", nargs="?", default=0.2, type=float)

    args = parser.parse_args()
    model = GPKAN()
    main(args, model)
