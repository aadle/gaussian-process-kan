import jax
import jax.numpy as jnp

from flax import nnx
from itertools import pairwise

class MLP(nnx.Module):
    def __init__(
        self, din: int, dout: int, hidden_layers: list[int], *, rngs: nnx.Rngs
    ) -> None:
        self.layers = []
        layer_dims = [din, *hidden_layers, dout]

        for in_dim, out_dim in pairwise(layer_dims):
            self.layers.append(nnx.Linear(in_dim, out_dim, rngs=rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))

        return self.layers[-1](x)

def main():
    mlp = MLP(2, 1, [16, 16], rngs=nnx.Rngs(1))
    x_in = jnp.ones((1, 2))
    print(mlp(x_in))

if __name__ == "__main__":
    main()
