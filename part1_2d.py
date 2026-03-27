import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def set_seed(seed_num=42):
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)


def mse_to_psnr(mse_val: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / mse_val)


def make_dir(folder_name: str):
    os.makedirs(folder_name, exist_ok=True)


class PositionalEncoding2D(nn.Module):
    def __init__(self, in_dim: int = 2, num_frequencies: int = 10):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32) * math.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def out_dim(self):
        return self.in_dim * (1 + 2 * self.num_frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        encoded_list = [coords]
        for freq in self.freq_bands:
            encoded_list.append(torch.sin(freq * coords))
            encoded_list.append(torch.cos(freq * coords))
        return torch.cat(encoded_list, dim=-1)


class ImageNeuralField(nn.Module):
    def __init__(self, pe_levels=10, hidden_width=256):
        super().__init__()
        self.pe = PositionalEncoding2D(in_dim=2, num_frequencies=pe_levels)
        input_dim = self.pe.out_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_width, 3),
            nn.Sigmoid()
        )

    def forward(self, uv_coords: torch.Tensor) -> torch.Tensor:
        uv_encoded = self.pe(uv_coords)
        rgb_pred = self.layers(uv_encoded)
        return rgb_pred


class ImageFittingDataset:
    def __init__(self, image_path: str, device: torch.device):
        pil_img = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_img).astype(np.float32) / 255.0

        self.image_tensor = torch.from_numpy(img_np).float().to(device)
        self.height, self.width = img_np.shape[:2]
        self.device = device

        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=device),
            torch.arange(self.width, device=device),
            indexing="ij"
        )

        uv_grid = torch.stack([xx, yy], dim=-1).float()
        uv_grid[..., 0] = uv_grid[..., 0] / self.width
        uv_grid[..., 1] = uv_grid[..., 1] / self.height

        self.all_uv = uv_grid.reshape(-1, 2)
        self.all_rgb = self.image_tensor.reshape(-1, 3)

    def sample_batch(self, batch_size: int):
        chosen_idx = torch.randint(0, self.all_uv.shape[0], (batch_size,), device=self.device)
        return self.all_uv[chosen_idx], self.all_rgb[chosen_idx]

    def get_all_uv(self):
        return self.all_uv


@torch.no_grad()
def render_full_image(model: nn.Module, dataset: ImageFittingDataset, chunk_size=65536):
    model.eval()
    uv_all = dataset.get_all_uv()

    rgb_chunks = []
    for start_idx in range(0, uv_all.shape[0], chunk_size):
        uv_chunk = uv_all[start_idx:start_idx + chunk_size]
        rgb_chunk = model(uv_chunk)
        rgb_chunks.append(rgb_chunk)

    rgb_flat = torch.cat(rgb_chunks, dim=0)
    rgb_img = rgb_flat.reshape(dataset.height, dataset.width, 3)
    return rgb_img


@torch.no_grad()
def render_image_to_numpy(model, dataset, chunk_size=65536):
    rgb_img = render_full_image(model, dataset, chunk_size=chunk_size)
    rgb_np = rgb_img.clamp(0, 1).detach().cpu().numpy()
    return rgb_np


def architecture_report_part1(pe_levels, hidden_width, learning_rate, num_iters, batch_size, save_path):
    report_text = []
    report_text.append("Part 1: 2D Neural Field Architecture Report")
    report_text.append("=========================================")
    report_text.append(f"Positional Encoding Levels (L): {pe_levels}")
    report_text.append(f"Hidden Width: {hidden_width}")
    report_text.append("Hidden Layers: 3")
    report_text.append("Output Activation: Sigmoid")
    report_text.append(f"Learning Rate: {learning_rate}")
    report_text.append(f"Iterations: {num_iters}")
    report_text.append(f"Batch Size: {batch_size}")
    report_text.append("Loss Function: MSELoss")
    report_text.append("Optimizer: Adam")

    with open(save_path, "w") as file_obj:
        file_obj.write("\n".join(report_text))


def train_2d_field(
    image_path: str,
    output_dir: str = "outputs_part1",
    pe_levels: int = 10,
    hidden_width: int = 256,
    lr: float = 1e-2,
    num_iters: int = 3000,
    batch_size: int = 10000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    make_dir(output_dir)
    device = torch.device(device)

    dataset = ImageFittingDataset(image_path, device)
    model = ImageNeuralField(pe_levels=pe_levels, hidden_width=hidden_width).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    psnr_log = []
    save_steps = [1, 100, 500, 1000, 2000, num_iters]

    for step_id in range(1, num_iters + 1):
        model.train()

        batch_uv, batch_rgb = dataset.sample_batch(batch_size)
        pred_rgb = model(batch_uv)
        loss_val = criterion(pred_rgb, batch_rgb)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        psnr_val = mse_to_psnr(loss_val.detach()).item()
        psnr_log.append(psnr_val)

        if step_id % 100 == 0 or step_id == 1:
            print(f"[Part1] Iter {step_id:04d} | Loss: {loss_val.item():.6f} | PSNR: {psnr_val:.2f}")

        if step_id in save_steps:
            rendered_img = render_full_image(model, dataset)
            rendered_uint8 = (rendered_img.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(rendered_uint8).save(
                os.path.join(output_dir, f"reconstruction_{step_id:04d}.png")
            )

    plt.figure(figsize=(8, 5))
    plt.plot(psnr_log)
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("2D Neural Field PSNR")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "psnr_curve.png"))
    plt.close()

    architecture_report_part1(
        pe_levels=pe_levels,
        hidden_width=hidden_width,
        learning_rate=lr,
        num_iters=num_iters,
        batch_size=batch_size,
        save_path=os.path.join(output_dir, "architecture_report.txt")
    )

    torch.save(model.state_dict(), os.path.join(output_dir, "image_field_model.pt"))
    return model


def run_part1_comparison_grid(
    image_path,
    output_dir="outputs_part1_comparison",
    pe_values=(4, 10),
    width_values=(64, 256),
    lr=1e-2,
    num_iters=2000,
    batch_size=10000,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    from visualization_utils import ensure_dir, save_image_grid

    ensure_dir(output_dir)
    device = torch.device(device)

    rendered_images = []
    grid_titles = []

    for pe_level in pe_values:
        for hidden_width in width_values:
            print(f"Training config: PE={pe_level}, Width={hidden_width}")
            dataset = ImageFittingDataset(image_path, device)
            model = ImageNeuralField(pe_levels=pe_level, hidden_width=hidden_width).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            for _ in range(1, num_iters + 1):
                batch_uv, batch_rgb = dataset.sample_batch(batch_size)
                pred_rgb = model(batch_uv)
                loss_val = criterion(pred_rgb, batch_rgb)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

            final_img = render_image_to_numpy(model, dataset)
            rendered_images.append(final_img)
            grid_titles.append(f"PE={pe_level}, Width={hidden_width}")

    save_image_grid(
        image_list=rendered_images,
        title_list=grid_titles,
        save_path=os.path.join(output_dir, "part1_2x2_grid.png"),
        rows=2,
        cols=2
    )


def train_on_two_images_for_progression(
    image_paths,
    output_root="outputs_part1_two_images",
    pe_levels=10,
    hidden_width=256,
    lr=1e-2,
    num_iters=3000,
    batch_size=10000,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    make_dir(output_root)

    for img_path in image_paths:
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        single_out = os.path.join(output_root, image_name)
        train_2d_field(
            image_path=img_path,
            output_dir=single_out,
            pe_levels=pe_levels,
            hidden_width=hidden_width,
            lr=lr,
            num_iters=num_iters,
            batch_size=batch_size,
            device=device
        )


if __name__ == "__main__":
    set_seed(7)

    image_file = "lion_hw4.png"
    if os.path.exists(image_file):
        train_2d_field(image_file)
    else:
        print("Place your image as 'fox.png' in this folder.")