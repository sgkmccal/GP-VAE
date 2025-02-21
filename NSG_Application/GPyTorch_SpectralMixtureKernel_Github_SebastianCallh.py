"""Module with GP model definitions using various kernels."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gpytorch as gp
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import (
    SpectralMixtureKernel,
    RBFKernel,
    AdditiveKernel,
    PolynomialKernel,
)

import torch
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal
from torch.optim import AdamW
from tqdm import tqdm

import math
from typing import Callable

import matplotlib.pyplot as plt
import seaborn as sns

from os import path
from pathlib import Path
import matplotlib as mpl
import numpy as np


class GP(gp.models.ExactGP):
    def __init__(self, cov, train_x, train_y):
        super(GP, self).__init__(train_x, train_y, GaussianLikelihood())
        self.mean = gp.means.ConstantMean()
        self.cov = cov

    def forward(self, x):
        return MultivariateNormal(self.mean(x), self.cov(x))

    def predict(self, x):
        self.eval()
        with torch.no_grad(), gp.settings.fast_pred_var():
            pred = self.likelihood(self(x))
            lower, upper = pred.confidence_region()

        return pred.mean, lower, upper

    def spectral_density(self, smk) -> MixtureSameFamily:
        """Returns the Mixture of Gaussians thet model the spectral density
        of the provided spectral mixture kernel."""
        mus = smk.mixture_means.detach().reshape(-1, 1)
        sigmas = smk.mixture_scales.detach().reshape(-1, 1)
        mix = Categorical(smk.mixture_weights.detach())
        comp = Independent(Normal(mus, sigmas), 1)
        return MixtureSameFamily(mix, comp)


class SMKernelGP(GP):
    def __init__(self, train_x, train_y, num_mixtures=10):
        kernel = SpectralMixtureKernel(num_mixtures)
        kernel.initialize_from_data(train_x, train_y)

        super(SMKernelGP, self).__init__(kernel, train_x, train_y)
        self.mean = gp.means.ConstantMean()
        self.cov = kernel

    def spectral_density(self):
        return super().spectral_density(self.cov)


class CompositeKernelGP(GP):
    def __init__(self, train_x, train_y, num_mixtures=10):
        smk = SpectralMixtureKernel(num_mixtures)
        smk.initialize_from_data(train_x, train_y)
        kernel = AdditiveKernel(
            smk,
            PolynomialKernel(2),
            RBFKernel(),
        )
        super(CompositeKernelGP, self).__init__(kernel, train_x, train_y)
        self.mean = gp.means.ConstantMean()
        self.smk = smk

    def spectral_density(self):
        return super().spectral_density(self.smk)
    

def train(
    model: GP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_iters: int,
    lr: float = 0.1,
    show_progress: bool = True,
):
    """Trains the provided model by maximising the marginal likelihood."""
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    loss = 0
    iterator = (
        tqdm(range(num_iters), desc="Epoch") if show_progress else range(num_iters)
    )

    for _ in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if show_progress:
            iterator.set_postfix(loss=loss.item())

    return loss.detach().cpu().item()


def train_with_restarts(
    make_model: Callable[[], GP],
    num_iters: int,
    num_restarts: int = 5,
    **kwargs,
) -> GP:
    """Trains the provided model by maximising the marginal likelihood.
    Performs several restarts and returns the best model to avoid bad local minima.
    """
    best_loss = math.inf
    best_model = None
    for _ in range(num_restarts):
        model = make_model()
        loss = train(
            model,
            model.train_inputs[0],
            model.train_targets,
            num_iters,
            **kwargs,
        )

        if loss < best_loss:
            best_loss = loss
            best_model = model

    return best_model.cpu(), best_loss


font = {"family": "DejaVu Sans", "size": 18}
mpl.rc("font", **font)

ROOT_DIR = Path(path.dirname(path.abspath(__file__)))
PLOTS_DIR = ROOT_DIR / "plots"


def save_plot(fig, name: str, format: str = "svg") -> None:
    fig.savefig(PLOTS_DIR / (name + f".{format}"), format=format)


def plot_cov_mat(kernel, ax, xx):
    ax.matshow(kernel(xx, xx).numpy())


def plot_kernel(kernel, ax, xx=torch.linspace(-0.1, 0.1, 1000), col="tab:blue"):
    x0 = torch.zeros(xx.size(0))
    ax.plot(xx.numpy(), np.diag(kernel(xx, x0).numpy()), color=col)


def plot_density(freq, density, ax):
    x = freq.numpy().flatten()
    y = density.numpy().flatten()
    ax.plot(x, y, color="tab:blue", lw=3)
    ax.fill_between(x, y, np.ones_like(x) * y.min(), color="tab:blue", alpha=0.5)
    ax.set_title("Kernel spectral density")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Log Density")


def plot_components(smk, ax, nyquist):

    iter = zip(
        smk.mixture_weights.detach(),
        smk.mixture_means.detach(),
        smk.mixture_scales.detach(),
    )

    freqs = torch.linspace(0, nyquist, 1000)
    for w, mu, sigma in iter:
        dist = Normal(mu % nyquist, sigma)
        ax.plot(
            freqs,
            w * dist.log_prob(freqs).flatten().exp(),
            label=f"Compoment at {dist.mean.item():.3}",
        )

    ax.set_title("'Kernel spectral density")
    ax.legend(fontsize=11)
    return ax


def plot_kernel(kernel, ax, xx=torch.linspace(-2, 2, 1000), col=sns.color_palette()[0]):
    x0 = torch.zeros(xx.size(0))
    ax.plot(xx.numpy(), np.diag(kernel(xx, x0).numpy()), lw=3, color=col)


fig, axs = plt.subplots(1, 3, figsize=(16, 6))
kernels = [
    gp.kernels.RBFKernel(),
    gp.kernels.CosineKernel(),
    gp.kernels.MaternKernel(1 / 2),
]
colors = [sns.color_palette()[i] for i in range(3)]

titles = ["RBF", "Cosine", "Matérn 1/2"]
n = 100
x0 = torch.zeros(n)
xx = torch.linspace(-2, 2, n)
for k, title, col, ax in zip(kernels, titles, colors, axs):
    plot_kernel(kernel=k, ax=ax, col=col)
    ax.set_title(title)
    ax.set_ylabel("Similarity")
    ax.set_xlabel("Distance")

fig.tight_layout()
save_plot(fig, "example_kernels")

