import jax.numpy as jnp
from gpkanmodel.test_functions import himmelblau, goldstein_price, trig
import pandas as pd
from jaxtyping import Array
from sklearn.model_selection import train_test_split


def data_setup(fn_name:str, n_samples:int) -> [Array | None, Array | None]:
    X, y = None, None
    match fn_name:
        case "himmelblau":
            x1, x2 = jnp.meshgrid(
                jnp.linspace(-5, 5, n_samples),
                jnp.linspace(-5, 5, n_samples),
            )
            X = jnp.stack([x1.flatten(), x2.flatten()]).T
            y = himmelblau(X[:, 0], X[:, 1]).reshape(-1, 1)

        case "goldstein":
            x1, x2 = jnp.meshgrid(
                jnp.linspace(-2, 2, n_samples),
                jnp.linspace(-2, 2, n_samples),
            )
            X = jnp.stack([x1.flatten(), x2.flatten()]).T
            y = goldstein_price(X[:, 0], X[:, 1]).reshape(-1, 1)

        case "trig":
            x1, x2 = jnp.meshgrid(
                jnp.linspace(0, 5, n_samples),
                jnp.linspace(0, 5, n_samples),
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
                (trollveggen[:, 0] >= x1_min)
                & (trollveggen[:, 0] <= x1_max)
                & (trollveggen[:, 1] >= x2_min)
                & (trollveggen[:, 1] <= x2_max)
            ]

            x1 = jnp.sort(jnp.unique(filtered_trollveggen[:, 0]))
            x2 = jnp.sort(
                jnp.unique(filtered_trollveggen[:, 1]), descending=True
            )
            X = filtered_trollveggen[:, :2]
            y = filtered_trollveggen[:, 2].reshape(-1, 1)

        case "grandcanyon":
            raise NotImplementedError(
                "'grandcanyon' is not implemented yet."
            )

    return x1, x2, X, y


def train_test_val_split(X, y, validation_size=0.2, seed=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    if validation_size:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=seed
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test


def standardize_data(val, mean=None, std=None):
    if mean is None and std is None:
        std = jnp.std(val, 0, keepdims=True)
        std = jnp.where(std == 0, 1.0, std)
        mean = jnp.mean(val, 0, keepdims=True)
    return (val - mean) / std, mean, std
