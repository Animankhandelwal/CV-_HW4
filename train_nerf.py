import os
import torch
import torch.nn.functional as F

from dataset_3d import RayData
from model import NeRFModel
from rendering import predict_radiance, render_image
from deliverables import (
    save_tensor_rgb_image,
    generate_ray_sample_visualization,
    render_all_test_frames,
    render_depth_video_frames,
    save_training_loss_curve,
    save_validation_psnr_curve,
    create_orbit_gif_from_frames,
)


def make_dir(folder_name):
    os.makedirs(folder_name, exist_ok=True)


def mse_to_psnr(mse_val):
    return 10.0 * torch.log10(1.0 / mse_val)


@torch.no_grad()
def evaluate_validation(model, ray_data, near=2.0, far=6.0, n_samples=32):
    if ray_data.images_val is None or ray_data.c2ws_val is None:
        return -1.0

    psnr_list = []
    for val_idx in range(ray_data.images_val.shape[0]):
        gt_img = ray_data.images_val[val_idx]
        pose = ray_data.c2ws_val[val_idx]

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

        mse_val = F.mse_loss(pred_img, gt_img)
        psnr_list.append(mse_to_psnr(mse_val).item())

    return float(sum(psnr_list) / len(psnr_list))


def train_nerf_main(
    npz_path="lego_200x200.npz",
    output_dir="outputs_part2",
    xyz_L=10,
    dir_L=4,
    hidden_dim=256,
    lr=5e-4,
    num_iters=1000,
    rays_per_batch=2048,
    near=2.0,
    far=6.0,
    n_samples=16,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    make_dir(output_dir)
    make_dir(os.path.join(output_dir, "intermediate_renders"))
    make_dir(os.path.join(output_dir, "orbit_frames"))
    make_dir(os.path.join(output_dir, "depth_frames"))

    device = torch.device(device)

    ray_data = RayData(npz_path=npz_path, device=device)
    model = NeRFModel(xyz_L=xyz_L, dir_L=dir_L, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_psnr_log = []
    train_loss_log = []
    val_psnr_steps = []
    val_psnr_values = []

    generate_ray_sample_visualization(
        ray_data=ray_data,
        save_path=os.path.join(output_dir, "ray_sample_visualization.png"),
        num_rays=100,
        near=near,
        far=far,
        n_samples=n_samples
    )

    for step_id in range(1, num_iters + 1):
        model.train()

        batch = ray_data.sample_rays(rays_per_batch)

        pred_rgb, _, _ = predict_radiance(
            model=model,
            ray_origins=batch.ray_origins,
            ray_directions=batch.ray_directions,
            near=near,
            far=far,
            n_samples=n_samples,
            perturb=True
        )

        loss_val = F.mse_loss(pred_rgb, batch.target_rgb)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss_log.append(loss_val.item())
        train_psnr_log.append(mse_to_psnr(loss_val.detach()).item())

        if step_id % 50 == 0 or step_id == 1:
            print(f"[NeRF] Iter {step_id:05d} | Loss: {loss_val.item():.6f} | Train PSNR: {train_psnr_log[-1]:.2f}")

        if step_id % 250 == 0:
            pred_train_view = render_image(
                model=model,
                c2w=ray_data.c2ws_train[0],
                focal=ray_data.focal,
                image_height=ray_data.height,
                image_width=ray_data.width,
                device=device,
                near=near,
                far=far,
                n_samples=n_samples
            )
            save_tensor_rgb_image(
                pred_train_view,
                os.path.join(output_dir, "intermediate_renders", f"render_{step_id:05d}.png")
            )

            val_psnr = evaluate_validation(
                model=model,
                ray_data=ray_data,
                near=near,
                far=far,
                n_samples=n_samples
            )
            if val_psnr > 0:
                val_psnr_steps.append(step_id)
                val_psnr_values.append(val_psnr)
                print(f"          Val PSNR: {val_psnr:.2f}")

    torch.save(model.state_dict(), os.path.join(output_dir, "nerf_model.pt"))

    save_training_loss_curve(
        loss_values=train_loss_log,
        save_path=os.path.join(output_dir, "training_loss_curve.png")
    )

    if len(val_psnr_steps) > 0:
        save_validation_psnr_curve(
            step_values=val_psnr_steps,
            psnr_values=val_psnr_values,
            save_path=os.path.join(output_dir, "validation_psnr_curve.png")
        )

    render_all_test_frames(
        model=model,
        ray_data=ray_data,
        output_folder=os.path.join(output_dir, "orbit_frames"),
        near=near,
        far=far,
        n_samples=n_samples
    )

    if os.path.exists(os.path.join(output_dir, "orbit_frames")) and len(os.listdir(os.path.join(output_dir, "orbit_frames"))) > 0:
        create_orbit_gif_from_frames(
            frames_folder=os.path.join(output_dir, "orbit_frames"),
            gif_path=os.path.join(output_dir, "orbit_render.gif"),
            fps=12
        )

    render_depth_video_frames(
        model=model,
        ray_data=ray_data,
        output_folder=os.path.join(output_dir, "depth_frames"),
        near=near,
        far=far,
        n_samples=n_samples
    )

    if os.path.exists(os.path.join(output_dir, "depth_frames")) and len(os.listdir(os.path.join(output_dir, "depth_frames"))) > 0:
        create_orbit_gif_from_frames(
            frames_folder=os.path.join(output_dir, "depth_frames"),
            gif_path=os.path.join(output_dir, "depth_render.gif"),
            fps=12
        )

    print("Finished training and deliverable generation.")


if __name__ == "__main__":
    train_nerf_main()