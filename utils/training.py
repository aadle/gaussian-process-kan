import jax
import jax.numpy as jnp


def loss_ll(y_true, mean, covariance):
    """
    Objective/loss function log-likelihood.

    How likely is it that our approximated parameters 'mean' and 'covariance'
    can produce the true targets 'y_true'?

    Parameters:
    y_true (jax.numpy.array): Training targets
    mean (jax.numpy.array): Pointwise mean approximated from predicted outputs
    covariance (jax.numpy.array): Covariance approximated from predicted outputs

    Returns:
    jax.numpy.array: log-likelihood of 'y_true', 'mean' and 'covariance'."""

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
    Simple learning rate scheduler reducing the learning rate over epochs.

    Parameters:
    epoch (int): Current epoch
    initial_lr (int): Starting point of learning rate

    Returns:
    int: New learning rate determined from the current epoch

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
