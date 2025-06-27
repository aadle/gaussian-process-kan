import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from jaxtyping import install_import_hook
from flax import nnx
from textwrap import wrap

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    transform,
)

# ---- Setup ----
plt.style.use(["science", "grid"])
jax.config.update("jax_enable_x64", True)


class GPKAN:
    def __init__(
        self,
        layers=[2, 5, 1, 1],
        n_grid_points=15,
        grid_min=0,
        grid_max=1,
        seed=42,
        parameter_transform=False,
        init_paramters=[
            1.0,
            1.0,
        ],  # signal variance \sigma_f^2, and length scale
        obs_stddev=1.0,  # noise variance \sigma_n^2
    ):
        self.layers = layers
        self.key = jr.key(seed)
        self.parameter_transform = parameter_transform

        # With the assumption that we will normalize our output, we set it to min 0 and max 1
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.n_grid_points = n_grid_points

        self.latent_grids = []
        self.latent_supports = []

        self.kernel_parameters = []
        self.init_parameters = init_paramters

        # ---------- GPJax parameters: ----------
        # With the assumption that we initialize all the GPs being the same, we
        # can define their paramaters the same way

        self.obs_stddev = obs_stddev
        self._kernel = gpx.kernels.RBF(
            # variance=0.54132485, lengthscale=0.54132485
            variance=self.init_parameters[1],
            lengthscale=self.init_parameters[0],
        )
        self._mean_function = gpx.mean_functions.Zero()
        self._likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=self.n_grid_points, obs_stddev=self.obs_stddev
        )
        self._prior = gpx.gps.Prior(
            mean_function=self._mean_function, kernel=self._kernel
        )
        self.posterior = self._prior * self._likelihood
        self.graphdef, self.params, self.others = nnx.split(
            self.posterior, Parameter, ...
        )

        self.init_model()

        if self.parameter_transform:
            self.transform_parameters(invert=True)

    # TODO:
    def init_model(self):
        for nin, nout in zip(self.layers[:-1], self.layers[1:]):
            self.key, subkey = jr.split(self.key, num=2)

            # Grid positions (latent x-values)
            latent_grid = jnp.linspace(
                self.grid_min, self.grid_max, self.n_grid_points
            ).reshape(-1, 1)
            latent_grids = jnp.tile(latent_grid, (nin, nout, 1, 1))
            self.latent_grids.append(latent_grids)

            # Grid values (latent y-values)
            # Xavier initialization idea
            xavier_init = jnp.sqrt(6 / (nin + nout))
            latent_supports = jr.uniform(
                subkey,
                shape=latent_grids.shape,
                minval=-xavier_init,
                maxval=xavier_init,
            )

            # latent_supports = (
            #     jr.normal(subkey, shape=latent_grids.shape) * 0.1 + 0
            # )

            self.latent_supports.append(latent_supports)

            # Kernel parameters
            kernel_parameters = [
                [self.params for _ in range(nout)] for _ in range(nin)
            ]
            self.kernel_parameters.append(kernel_parameters)

        print("Model initialized.")

    def transform_parameters(self, invert=True):
        for layer_idx in range(len(self.kernel_parameters)):
            layer = self.kernel_parameters[layer_idx]
            for input_idx in range(len(layer)):
                parameters = layer[input_idx]
                for parameter_idx in range(len(parameters)):
                    self.kernel_parameters[layer_idx][input_idx][
                        parameter_idx
                    ] = transform(
                        self.kernel_parameters[layer_idx][input_idx][
                            parameter_idx
                        ],
                        DEFAULT_BIJECTION,
                        inverse=invert,
                    )

    # TODO:
    def predictive_posterior(
        self,
        X_latent,
        y_latent,
        X_in,
        gp_parameters,
    ):
        D_latent = gpx.Dataset(X=X_latent, y=y_latent)
        parameters = gp_parameters
        posterior = nnx.merge(self.graphdef, parameters, *self.others)

        latent_dist = posterior.predict(
            X_in, train_data=D_latent
        )  # gives covariance matrix without observational standard variance $\sigma_n^2$ on the diagonal
        # pred_dist = posterior.predict(X_in, train_data=D_latent)
        pred_dist = posterior.likelihood(
            latent_dist
        )  # gives covariance matrix with observational standard variance $\sigma_n^2$ on the diagonal

        return pred_dist

    def sample(
        self,
        Xs_latent,
        ys_latent,
        X_test,
        kernel_parameters,
        key=None,
    ):
        if key is None:
            key = self.key

        act = X_test

        for layer_idx, (nin, nout) in enumerate(  # Iterate over layers
            zip(self.layers[:-1], self.layers[1:])
        ):
            out_samples = jnp.zeros(
                (act.shape[0], nout)
            )  # Initialize a vector for the post-activations
            keys_needed = nin * nout
            key, *all_keys = jr.split(key, num=keys_needed + 1)
            all_keys = jnp.array(all_keys).reshape(nin, nout)

            for nin_idx in range(nin):  # nin loop
                act_in = act[:, nin_idx].reshape(-1, 1)
                for nout_idx in range(nout):  # nout loop
                    sample_key = all_keys[nin_idx, nout_idx]

                    X_latent = Xs_latent[layer_idx][nin_idx][nout_idx]
                    y_latent = ys_latent[layer_idx][nin_idx][nout_idx]
                    kernel_parameter = kernel_parameters[layer_idx][nin_idx][
                        nout_idx
                    ]

                    posterior = self.predictive_posterior(
                        X_latent, y_latent, act_in, kernel_parameter
                    )
                    posterior_sample = posterior.sample(
                        sample_shape=(), seed=sample_key
                    ).flatten()
                    out_samples = out_samples.at[:, nout_idx].add(
                        posterior_sample
                    )
                    # print(posterior_sample.shape)

            act = out_samples

        return act

    def sample_statistics(
        self,
        Xs_latent,
        ys_latent,
        X_test,
        kernel_parameters,
        n_samples=10,
        key=None,
    ):
        if key is None:
            key = self.key

        key, *keys = jr.split(key, num=n_samples + 1)
        sampler = lambda sample_key: self.sample(
            Xs_latent, ys_latent, X_test, kernel_parameters, key=sample_key
        )
        batched_sampler = jax.jit(jax.vmap(sampler))
        samples = batched_sampler(jnp.array(keys)).squeeze()

        if samples.ndim == 1:
            samples = samples[:, None]

        mu = jnp.mean(samples, axis=0)
        # print(mu.shape)
        sigma = jnp.var(samples, axis=0) * jnp.eye(mu.shape[0])
        # sigma = jnp.cov(samples, rowvar=False) # covariance
        return mu, sigma

    def plot_neurons(
        self,
        save_fig=False,
        save_path="figs/test_fig",
    ):
        for layer_idx, (nin, nout) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            subplot_size = 3
            min_width = 14

            # Determine figure size
            if nout == 1:
                figsize = (min_width, nin * subplot_size)
            elif nin == 1:
                figsize = (nout * subplot_size, subplot_size)
            else:
                figsize = (nout * subplot_size, nin * subplot_size)

            # Create subplots
            if nout == 1:
                fig, axs = plt.subplots(nrows=nin, ncols=1, figsize=figsize)
                axs = np.array(axs).reshape(nin, 1)
            else:
                fig, axs = plt.subplots(nrows=nin, ncols=nout, figsize=figsize)
                axs = np.array(axs).reshape(nin, nout)

            for nin_idx in range(nin):
                for nout_idx in range(nout):
                    ax = axs[nin_idx, nout_idx]

                    X_latent = self.latent_grids[layer_idx][nin_idx][nout_idx]
                    y_latent = self.latent_supports[layer_idx][nin_idx][
                        nout_idx
                    ]
                    x_min, x_max = jnp.min(X_latent), jnp.max(X_latent)
                    x_range = x_max - x_min
                    X = jnp.linspace(
                        x_min - 0.1 * x_range, x_max + 0.1 * x_range, 100
                    ).reshape(-1, 1)

                    kernel_parameter = self.kernel_parameters[layer_idx][
                        nin_idx
                    ][nout_idx]
                    length_scale = kernel_parameter["prior"]["kernel"][
                        "lengthscale"
                    ].value
                    signal_variance = kernel_parameter["prior"]["kernel"][
                        "variance"
                    ].value
                    noise_obs_stddev = kernel_parameter["likelihood"][
                        "obs_stddev"
                    ].value

                    posterior = self.predictive_posterior(
                        X_latent, y_latent, X, kernel_parameter
                    )
                    posterior_mean = posterior.mean()
                    posterior_stddev = posterior.stddev()

                    ax.set_title(
                        "\n".join(
                            wrap(
                                f"$l$: {length_scale:.2f}, $\sigma_f^2$: {signal_variance:.2f}, $\sigma_n$: {noise_obs_stddev:.2f}"
                            )
                        ),
                        fontsize=12,
                    )

                    ax.plot(
                        X,
                        posterior_mean,
                        linestyle="--",
                        linewidth=0.7,
                        color="black",
                    )
                    ax.scatter(X_latent, y_latent, s=30, color="tab:orange")
                    ax.fill_between(
                        X.flatten(),
                        posterior_mean + 2 * posterior_stddev,
                        posterior_mean - 2 * posterior_stddev,
                        alpha=0.15,
                        color="tab:blue",
                    )

                    ax.set_xlabel("h", fontsize=10)
                    ax.set_ylabel("z", fontsize=10)
                    ax.set_box_aspect(1)

                    # Add Input dimension labels
                    if nout == 1 or nout_idx == nout - 1:
                        ax2 = ax.twinx()
                        ax2.set_ylabel(
                            f"Input dimension {nin_idx + 1}", fontsize=12
                        )
                        ax2.set_yticks([])
                        ax2.set_box_aspect(1)

            fig.suptitle(f"Layer {layer_idx + 1}", fontsize=25)

            if nout == 1:
                # Manual spacing for vertical column
                fig.subplots_adjust(top=0.90, hspace=0.6)
            else:
                plt.tight_layout(rect=[0, 0, 1, 0.95])

            if save_fig:
                plt.savefig(save_path + f"layer_{layer_idx + 1}.png", dpi=500)


if __name__ == "__main__":
    model = GPKAN(
        layers=[2, 5, 3, 1],
        n_grid_points=10,
        grid_min=0,
        grid_max=5,
        parameter_transform=True,
    )
    model.plot_neurons(save_fig=True)
    plt.show()

    # f = lambda x, y: jnp.sin(x) ** 10 + jnp.cos(10 + y * x) * jnp.cos(x)

    # x_limit = 5.0
    # samples = 10
    # n_samples = samples**2
    # x1, x2 = jnp.meshgrid(
    #     jnp.linspace(0.0, x_limit, samples), jnp.linspace(0.0, x_limit, samples)
    # )
    # X = jnp.stack([x1.flatten(), x2.flatten()]).T
    # y = f(X[:, 0], X[:, 1]).reshape(-1, 1)

    # model = GPKAN()
    # # model.transform_parameters()
    # q = model.sample(
    #     model.latent_grids, model.latent_supports, X, model.kernel_parameters
    # )
    # print(q)
