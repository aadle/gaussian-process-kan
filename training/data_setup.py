import jax
import jax.numpy as jnp
from gpkanmodel.test_functions import himmelblau, goldstein_price, trig

def data_setup(args) -> [jax.Array | None, jax.Array | None]:
    X, y = None, None
    match args.function:
        case "himmelblau":
            x1, x2 = jnp.meshgrid(
                jnp.linspace(-5, 5, args.num_samples), 
                jnp.linspace(-5, 5, args.num_samples)
            )
            X = jnp.stack([x1.flatten(), x2.flatten()]).T
            y = himmelblau(X[:, 0], X[:, 1]).reshape(-1, 1)

        case "goldstein":
            x1, x2 = jnp.meshgrid(
                jnp.linspace(-2, 2, args.num_samples), 
                jnp.linspace(-2, 2, args.num_samples)
            )
            X = jnp.stack([x1.flatten(), x2.flatten()]).T
            y = goldstein_price(X[:, 0], X[:, 1]).reshape(-1, 1)

        case "trig":
            x1, x2 = jnp.meshgrid(
                jnp.linspace(0, 5, args.num_samples), 
                jnp.linspace(0, 5, args.num_samples)
            )
            X = jnp.stack([x1.flatten(), x2.flatten()]).T
            y = trig(X[:, 0], X[:, 1]).reshape(-1, 1)

        case "trollveggen":
            pass

        case "grandcanyon":
            pass
    return X, y
