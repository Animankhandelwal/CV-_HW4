import numpy as np
import torch
from dataclasses import dataclass


def load_data(npz_path: str):
    raw_data = np.load(npz_path)

    output_data = {
        "images_train": raw_data["images_train"],
        "c2ws_train": raw_data["c2ws_train"],
        "focal": float(raw_data["focal"]),
    }

    if "images_val" in raw_data:
        output_data["images_val"] = raw_data["images_val"]
    if "c2ws_val" in raw_data:
        output_data["c2ws_val"] = raw_data["c2ws_val"]
    if "c2ws_test" in raw_data:
        output_data["c2ws_test"] = raw_data["c2ws_test"]

    return output_data


def pixel_to_camera(uv_coords: torch.Tensor, focal: float, image_width: int, image_height: int):
    principal_x = image_width / 2.0
    principal_y = image_height / 2.0

    u_val = uv_coords[..., 0]
    v_val = uv_coords[..., 1]

    x_cam = (u_val - principal_x) / focal
    y_cam = (v_val - principal_y) / focal
    z_cam = torch.ones_like(x_cam)

    xyz_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    return xyz_cam


def camera_to_world(xyz_cam: torch.Tensor, c2w: torch.Tensor):
    prefix_shape = xyz_cam.shape[:-1]
    xyz_cam_h = torch.cat(
        [xyz_cam, torch.ones(*prefix_shape, 1, device=xyz_cam.device)],
        dim=-1
    )
    xyz_world_h = xyz_cam_h @ c2w.T
    xyz_world = xyz_world_h[..., :3]
    return xyz_world


def pixels_to_rays(uv_coords: torch.Tensor, c2w: torch.Tensor, focal: float, image_width: int, image_height: int):
    cam_pts = pixel_to_camera(uv_coords, focal, image_width, image_height)
    world_pts = camera_to_world(cam_pts, c2w)

    cam_origin = c2w[:3, 3]
    ray_origins = cam_origin.unsqueeze(0).expand_as(world_pts)
    ray_dirs = world_pts - ray_origins
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    return ray_origins, ray_dirs


@dataclass
class RaysBatch:
    ray_origins: torch.Tensor
    ray_directions: torch.Tensor
    target_rgb: torch.Tensor


class RayData:
    def __init__(self, npz_path: str, device: torch.device):
        loaded = load_data(npz_path)

        self.device = device
        self.focal = loaded["focal"]

        self.images_train = loaded["images_train"].astype(np.float32)
        if self.images_train.max() > 1.0:
            self.images_train = self.images_train / 255.0

        self.c2ws_train = loaded["c2ws_train"].astype(np.float32)

        self.images_train = torch.from_numpy(self.images_train).float().to(device)
        self.c2ws_train = torch.from_numpy(self.c2ws_train).float().to(device)

        self.images_val = None
        self.c2ws_val = None
        self.c2ws_test = None

        if "images_val" in loaded:
            img_val = loaded["images_val"].astype(np.float32)
            if img_val.max() > 1.0:
                img_val = img_val / 255.0
            self.images_val = torch.from_numpy(img_val).float().to(device)

        if "c2ws_val" in loaded:
            self.c2ws_val = torch.from_numpy(loaded["c2ws_val"].astype(np.float32)).float().to(device)

        if "c2ws_test" in loaded:
            self.c2ws_test = torch.from_numpy(loaded["c2ws_test"].astype(np.float32)).float().to(device)

        self.num_train, self.height, self.width, _ = self.images_train.shape
        self._precompute_all_train_rays()

    def _precompute_all_train_rays(self):
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing="ij"
        )

        uv_grid = torch.stack([xx, yy], dim=-1).float() + 0.5
        uv_flat = uv_grid.reshape(-1, 2)

        origin_list = []
        direction_list = []
        color_list = []

        for img_id in range(self.num_train):
            pose_mat = self.c2ws_train[img_id]
            rgb_flat = self.images_train[img_id].reshape(-1, 3)

            ray_o, ray_d = pixels_to_rays(
                uv_coords=uv_flat,
                c2w=pose_mat,
                focal=self.focal,
                image_width=self.width,
                image_height=self.height
            )

            origin_list.append(ray_o)
            direction_list.append(ray_d)
            color_list.append(rgb_flat)

        self.all_ray_origins = torch.cat(origin_list, dim=0)
        self.all_ray_directions = torch.cat(direction_list, dim=0)
        self.all_ray_colors = torch.cat(color_list, dim=0)

    def sample_rays(self, num_rays: int):
        total_rays = self.all_ray_origins.shape[0]
        chosen_idx = torch.randint(0, total_rays, (num_rays,), device=self.device)

        return RaysBatch(
            ray_origins=self.all_ray_origins[chosen_idx],
            ray_directions=self.all_ray_directions[chosen_idx],
            target_rgb=self.all_ray_colors[chosen_idx],
        )