import jax
import jax.numpy as jnp
from gpkanmodel.test_functions import himmelblau, goldstein_price, trig
import pandas as pd

def data_setup(args) -> [jax.Array | None, jax.Array | None]:
    X, y = None, None
    match args.function:
        case "himmelblau":
            x1, x2 = jnp.meshgrid(
                jnp.linspace(-5, 5, args.n_samples), 
                jnp.linspace(-5, 5, args.n_samples)
            )
            X = jnp.stack([x1.flatten(), x2.flatten()]).T
            y = himmelblau(X[:, 0], X[:, 1]).reshape(-1, 1)

        case "goldstein":
            x1, x2 = jnp.meshgrid(
                jnp.linspace(-2, 2, args.n_samples), 
                jnp.linspace(-2, 2, args.n_samples)
            )
            X = jnp.stack([x1.flatten(), x2.flatten()]).T
            y = goldstein_price(X[:, 0], X[:, 1]).reshape(-1, 1)

        case "trig":
            x1, x2 = jnp.meshgrid(
                jnp.linspace(0, 5, args.n_samples), 
                jnp.linspace(0, 5, args.n_samples)
            )
            X = jnp.stack([x1.flatten(), x2.flatten()]).T
            y = trig(X[:, 0], X[:, 1]).reshape(-1, 1)

        case "trollveggen":
            df_trollveggen = pd.read_csv("../datasets/troll3.csv")
            trollveggen = jnp.array(df_trollveggen.values)

            # Limit the dataset
            x1_min, x1_max = 7.55, 7.96
            x2_min, x2_max = 62.3, 62.66875
            filtered_trollveggen = trollveggen[
                (trollveggen[:, 0] >= x1_min) & (trollveggen[:, 0] <= x1_max) &
                (trollveggen[:, 1] >= x2_min) & (trollveggen[:, 1] <= x2_max)
            ]

            x1 = jnp.sort(jnp.unique(filtered_trollveggen[:, 0]))  
            x2 = jnp.sort(jnp.unique(filtered_trollveggen[:, 1]), descending=True)
            X = filtered_trollveggen[:, :2]
            y = filtered_trollveggen[:, 2].reshape(-1, 1)

        case "grandcanyon":
           raise NotImplementedError(f"{args.function} is not implemented yet.") 
    return x1, x2, X, y
