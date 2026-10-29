from matplotlib import pyplot as plt
import torch
from SSM.SDEs.SDE import SDE
from matplotlib.animation import FuncAnimation


class Lorenz63(SDE):
    def __init__(self, config, device):
        super().__init__(dimension_state=3, dimension_noise=3)
        self.sigma = torch.tensor(config['sigma'], device=device, dtype=torch.float32)
        self.rho = torch.tensor(config['rho'], device=device, dtype=torch.float32)
        self.beta = torch.tensor(config['beta'], device=device, dtype=torch.float32)
        self.sigma_x = torch.tensor(config['sigma_x'], device=device, dtype=torch.float32)
        self.sigma_y = torch.tensor(config['sigma_y'], device=device, dtype=torch.float32)
        self.sigma_z = torch.tensor(config['sigma_z'], device=device, dtype=torch.float32)

    def drift(self, t, x):
        x_val, y_val, z_val = x[:, 0], x[:, 1], x[:, 2]
        dx = self.sigma * (y_val - x_val)
        dy = x_val * (self.rho - z_val) - y_val
        dz = x_val * y_val - self.beta * z_val
        return torch.stack([dx, dy, dz], dim=1)

    def diffusion(self, t, x):
        batch_size = x.shape[0]
        diff = torch.zeros(batch_size, 3, 3, device=x.device, dtype=x.dtype)
        diff[:, 0, 0] = self.sigma_x
        diff[:, 1, 1] = self.sigma_y
        diff[:, 2, 2] = self.sigma_z
        return diff

    def plot_paths(self, paths):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i in range(paths.shape[0]):
            x_path = paths[i, :, 0].cpu().numpy()
            y_path = paths[i, :, 1].cpu().numpy()
            z_path = paths[i, :, 2].cpu().numpy()
            ax.plot(x_path, y_path, z_path, color=colors[i % len(colors)], label=f'Realization {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{self.__class__.__name__.upper()} SDE Realizations')
        ax.legend()
        plt.savefig(f'{self.__class__.__name__.lower()}_realizations.png')
        print(f"3D plot saved as {self.__class__.__name__.lower()}_realizations.png")
