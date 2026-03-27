import os
import numpy as np
from PIL import Image

from rendering import render_image, render_depth_image, sample_along_rays
from visualization_utils import (
    ensure_dir,
    save_plot_curve,
    create_gif_from_folder,
    normalize_depth_for_display,
    visualize_rays_and_points,
)


def save_tensor_rgb_image(img_tensor, save_path):
    img_uint8 = (img_tensor.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(save_path)


def save_depth_image(depth_tensor, save_path):
    depth_np = depth_tensor.detach().cpu().numpy()
    depth_uint8 = normalize_depth_for_display(depth_np)
    Image.fromarray(depth_uint8).save(save_path)


def generate_ray_sample_visualization(ray_data, save_path, num_rays=100, near=2.0, far=6.0, n_samples=32):
    batch = ray_data.sample_rays(num_rays)
    sampled_xyz, _ = sample_along_rays(
        batch.ray_origins,
        batch.ray_directions,
        near=near,
        far=far,
        n_samples=n_samples,
        perturb=False
    )

    camera_origins = ray_data.c2ws_train[:, :3, 3].detach().cpu().numpy()
    ray_origins = batch.ray_origins.detach().cpu().numpy()
    ray_directions = batch.ray_directions.detach().cpu().numpy()
    sampled_points = sampled_xyz.detach().cpu().numpy()

    visualize_rays_and_points(
        camera_origins=camera_origins,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        sampled_points=sampled_points,
        save_path=save_path,
        max_rays=num_rays
    )


def render_all_test_frames(model, ray_data, output_folder, near=2.0, far=6.0, n_samples=32):
    ensure_dir(output_folder)

    if ray_data.c2ws_test is None:
        return

    for frame_idx in range(ray_data.c2ws_test.shape[0]):
        pose = ray_data.c2ws_test[frame_idx]
        pred_img = render_image(
            model=model,
            c2w=pose,
            focal=ray_data.focal,
            image_height=ray_data.height,
            image_width=ray_data.width,
            device=ray_data.device,
            near=near,
            far=far,
            n_samples=n_samples
        )
        save_tensor_rgb_image(pred_img, os.path.join(output_folder, f"frame_{frame_idx:04d}.png"))


def render_depth_video_frames(model, ray_data, output_folder, near=2.0, far=6.0, n_samples=32):
    ensure_dir(output_folder)

    if ray_data.c2ws_test is None:
        return

    for frame_idx in range(ray_data.c2ws_test.shape[0]):
        pose = ray_data.c2ws_test[frame_idx]
        depth_img = render_depth_image(
            model=model,
            c2w=pose,
            focal=ray_data.focal,
            image_height=ray_data.height,
            image_width=ray_data.width,
            device=ray_data.device,
            near=near,
            far=far,
            n_samples=n_samples
        )
        save_depth_image(depth_img, os.path.join(output_folder, f"depth_{frame_idx:04d}.png"))


def save_training_loss_curve(loss_values, save_path):
    save_plot_curve(
        y_values=loss_values,
        save_path=save_path,
        title="Training Loss Curve",
        xlabel="Iteration",
        ylabel="MSE Loss"
    )


def save_validation_psnr_curve(step_values, psnr_values, save_path):
    save_plot_curve(
        y_values=psnr_values,
        x_values=step_values,
        save_path=save_path,
        title="Validation PSNR Curve",
        xlabel="Iteration",
        ylabel="PSNR"
    )


def create_orbit_gif_from_frames(frames_folder, gif_path, fps=12):
    create_gif_from_folder(frames_folder, gif_path, fps=fps)