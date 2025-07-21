import jax
import jax.numpy as jnp


def loss_ll(y_true, mean, covariance):
    """
    Computes the negative log-likelihood loss for a multivariate Gaussian.

    Args:
        y_true (jax.numpy.array): True target values.
        mean (jax.numpy.array): Predicted mean values.
        covariance (jax.numpy.array): Predicted covariance matrix (assumed diagonal).

    Returns:
        jax.numpy.array: The negative log-likelihood of the targets under the predicted distribution.
    """
    diag_elements = jnp.diag(covariance)
    covariance_inv = jnp.diag(1.0 / diag_elements)
    log_det = jnp.sum(jnp.log(diag_elements))
    y_true = y_true.flatten()

    return -(
        -0.5
        * (
            y_true.shape[0] * jnp.log(2 * jnp.pi)
            + log_det
            + (y_true - mean).T @ covariance_inv @ (y_true - mean)
        )
    )


@jax.jit
def get_learning_rate(epoch, initial_lr=0.0001):
    """
    Computes the learning rate for a given epoch using exponential decay.

    Args:
        epoch (int): The current training epoch.
        initial_lr (float, optional): The initial learning rate. Defaults to 0.0001.

    Returns:
        float: The decayed learning rate for the current epoch.
    """
    return initial_lr * (0.95 ** (epoch // 50))


@jax.jit
def clip_gradients(grads, max_norm=1.0):
    """
    Clips the gradients to a maximum L2 norm.

    Args:
        grads: A PyTree of gradients (e.g., as returned by a JAX autodiff function).
        max_norm (float, optional): The maximum allowed L2 norm for the gradients. Defaults to 1.0.

    Returns:
        A PyTree of gradients with the same structure as `grads`, where the gradients are scaled
        down if their global L2 norm exceeds `max_norm`.
    """
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    clip_factor = jnp.minimum(1.0, max_norm / grad_norm)
    return jax.tree.map(lambda g: g * clip_factor, grads)
