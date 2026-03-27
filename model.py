import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim: int, num_frequencies: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies

        freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32) * math.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def output_dim(self):
        return self.input_dim * (1 + 2 * self.num_frequencies)

    def forward(self, x: torch.Tensor):
        encoded_parts = [x]
        for freq in self.freq_bands:
            encoded_parts.append(torch.sin(freq * x))
            encoded_parts.append(torch.cos(freq * x))
        return torch.cat(encoded_parts, dim=-1)


class NeRFModel(nn.Module):
    def __init__(self, xyz_L=10, dir_L=4, hidden_dim=256):
        super().__init__()

        self.xyz_pe = PositionalEncoding(input_dim=3, num_frequencies=xyz_L)
        self.dir_pe = PositionalEncoding(input_dim=3, num_frequencies=dir_L)

        xyz_dim = self.xyz_pe.output_dim
        dir_dim = self.dir_pe.output_dim

        self.fc1 = nn.Linear(xyz_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.fc5 = nn.Linear(hidden_dim + xyz_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)

        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.feature_fc = nn.Linear(hidden_dim, hidden_dim)

        self.rgb_fc1 = nn.Linear(hidden_dim + dir_dim, 128)
        self.rgb_fc2 = nn.Linear(128, 3)

    def forward(self, xyz: torch.Tensor, ray_d: torch.Tensor):
        xyz_encoded = self.xyz_pe(xyz)
        dir_encoded = self.dir_pe(ray_d)

        feat = F.relu(self.fc1(xyz_encoded))
        feat = F.relu(self.fc2(feat))
        feat = F.relu(self.fc3(feat))
        feat = F.relu(self.fc4(feat))

        feat = torch.cat([feat, xyz_encoded], dim=-1)
        feat = F.relu(self.fc5(feat))
        feat = F.relu(self.fc6(feat))
        feat = F.relu(self.fc7(feat))
        feat = F.relu(self.fc8(feat))

        sigma = F.relu(self.sigma_head(feat))

        color_feature = self.feature_fc(feat)
        rgb_input = torch.cat([color_feature, dir_encoded], dim=-1)
        rgb_hidden = F.relu(self.rgb_fc1(rgb_input))
        rgb = torch.sigmoid(self.rgb_fc2(rgb_hidden))

        return rgb, sigma