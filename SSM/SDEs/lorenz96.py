import torch
from matplotlib import pyplot as plt
from SSM.SDEs.SDE import SDE


class Lorenz96(SDE):
    def __init__(self, config, device):
        dimension_state = config["dimension_state"]
        super().__init__(dimension_state=dimension_state, dimension_noise=dimension_state)
        self.F = torch.tensor(config["F"], device=device, dtype=torch.float32)
        self.sigma = config["sigma"]

    def drift(self, t, x):
        x_ip1 = torch.roll(x, shifts=-1, dims=1)
        x_im1 = torch.roll(x, shifts=1, dims=1)
        x_im2 = torch.roll(x, shifts=2, dims=1)
        return (x_ip1 - x_im2) * x_im1 - x + self.F

    def diffusion(self, t, x):
        batch_size, dim = x.shape
        eye = torch.eye(dim, device=x.device, dtype=x.dtype) * self.sigma
        return eye.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def plot_paths(self, paths):
        # Plot the first realization with line + heatmap views.
        if paths.ndim == 3:
            path = paths[0]
        else:
            path = paths

        if isinstance(path, torch.Tensor):
            path = path.detach().cpu()

        n_steps, dim = path.shape
        t = torch.arange(n_steps)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Top-left: spatial snapshot at final time.
        axes[0, 0].plot(torch.arange(dim), path[-1].numpy(), color="tab:blue", lw=1.5)
        axes[0, 0].axhline(0.0, color="black", lw=1, ls="--", alpha=0.6)
        axes[0, 0].set_title(r"$x_k(t_{end})$")
        axes[0, 0].set_xlabel("k")
        axes[0, 0].set_ylabel("x")

        # Top-right: time series of a couple of components.
        axes[0, 1].plot(t.numpy(), path[:, 0].numpy(), color="tab:blue", lw=1.5, label="x_0(t)")
        if dim > 1:
            axes[0, 1].plot(t.numpy(), path[:, 1].numpy(), color="tab:orange", lw=1.0, label="x_1(t)")
        axes[0, 1].set_title("x_k(t)")
        axes[0, 1].set_xlabel("t")
        axes[0, 1].set_ylabel("x")
        axes[0, 1].legend(loc="upper left", fontsize=8, frameon=False)

        # Bottom-left: heatmap of x_k(t).
        im0 = axes[1, 0].imshow(
            path.T.numpy(),
            aspect="auto",
            origin="lower",
            cmap="viridis",
        )
        axes[1, 0].set_title(r"$x_k(t)$")
        axes[1, 0].set_xlabel("t")
        axes[1, 0].set_ylabel("k")
        fig.colorbar(im0, ax=axes[1, 0], orientation="horizontal", fraction=0.05, pad=0.12)

        # Bottom-right: heatmap of anomalies (demeaned over k).
        anom = path - path.mean(dim=1, keepdim=True)
        im1 = axes[1, 1].imshow(
            anom.T.numpy(),
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
        )
        axes[1, 1].set_title(r"$x_k(t)$ (anomaly)")
        axes[1, 1].set_xlabel("t")
        axes[1, 1].set_ylabel("k")
        fig.colorbar(im1, ax=axes[1, 1], orientation="horizontal", fraction=0.05, pad=0.12)

        fig.tight_layout()
        out_name = f"{self.__class__.__name__.lower()}_paths.png"
        plt.savefig(out_name, dpi=150)
        plt.close(fig)
        print(f"Plot saved as {out_name}")
