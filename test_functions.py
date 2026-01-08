import jax.numpy as jnp


def himmelblau(x, y):
    assert jnp.all((-5 <= x) & (x <= 5)) and jnp.all((-5 <= y) & (y <= 5)), (
        "x and y must be in the range [-5, 5]"
    )
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def goldstein_price(x, y):
    assert jnp.all((-2 <= x) & (x <= 2)) and jnp.all((-2 <= y) & (y <= 2)), (
        "x and y must be in the range [-2, 2]"
    )
    return (
        1
        + ((x + y + 1) ** 2)
        * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + ((2 * x - 3 * y) ** 2)
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )


def stack_overflow(x, y):
    """
    https://stackoverflow.com/questions/71111212/save-plot-from-matplotlib-so-that-plot-is-centered
    -3 <= x, y <= 3
    """
    assert jnp.all((-3 <= x) & (x <= 3)) and jnp.all((-3 <= y) & (y <= 3)), (
        "x and y must be in the range [-3, 3]"
    )
    return (1 - (x**2) / 2 + x**5 + y**3) * jnp.exp(-(x**2) - y**2)


def rosenbrock(x, y):
    assert jnp.all((-2 <= x) & (x <= 2)) and jnp.all((-1 <= y) & (y <= 3)), (
        "x and y must be in the range [-3, 3]"
    )
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def beale(x, y):
    assert jnp.all((-4.5 <= x) & (x <= 4.5)) and jnp.all(
        (-4.5 <= y) & (y <= 4.5)
    ), "x and y must be in the range [-4.5, 4.5]"
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def three_hump_camel(x, y):
    assert jnp.all((-5 <= x) & (x <= 5)) and jnp.all((-5 <= y) & (y <= 5)), (
        "x and y must be in the range [-5, 5]"
    )
    return 2 * x**2 - 1.05 * x**4 + (x**6) / 6 + x * y + y**2


def deep_gp_test_function(x, y):
    """
    Comes from https://jmlr.org/papers/v19/18-015.html
    """
    assert jnp.all((0 <= x) & (x <= 1)) and jnp.all((0 <= y) & (y <= 1)), (
        "x and y must be in the range [0, 1]"
    )
    # Start with the first term which applies everywhere
    result = jnp.cos(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)

    # Create masks for the different regions
    mask_1 = (x >= 1 / 4) & (x <= 3 / 4) & (y >= 1 / 4) & (y <= 3 / 4)
    mask_2 = (x >= 1 / 2) & (x <= 3 / 4) & (y >= 1 / 2) & (y <= 3 / 4)
    mask_3 = (x >= 1 / 4) & (x <= 1 / 2) & (y >= 1 / 4) & (y <= 1 / 2)

    # Add the region-specific terms
    result = result + jnp.sin(4 * jnp.pi * x) * jnp.sin(4 * jnp.pi * y) * mask_1
    result = result + jnp.sin(8 * jnp.pi * x) * jnp.sin(8 * jnp.pi * y) * mask_2
    result = (
        result + jnp.sin(16 * jnp.pi * x) * jnp.sin(16 * jnp.pi * y) * mask_3
    )

    return result
