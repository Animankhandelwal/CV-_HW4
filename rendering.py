import torch
from dataset_3d import pixels_to_rays


def sample_along_rays(ray_origins, ray_directions, near=2.0, far=6.0, n_samples=32, perturb=True):
    num_rays = ray_origins.shape[0]
    device = ray_origins.device

    t_vals = torch.linspace(near, far, n_samples, device=device)
    t_vals = t_vals.unsqueeze(0).expand(num_rays, n_samples)

    if perturb:
        step_width = (far - near) / n_samples
        t_vals = t_vals + torch.rand_like(t_vals) * step_width

    sample_xyz = ray_origins[:, None, :] + t_vals[..., None] * ray_directions[:, None, :]
    return sample_xyz, t_vals


def volume_render(rgb_samples, sigma_samples, t_vals):
    sigma_samples = sigma_samples.squeeze(-1)

    delta_vals = t_vals[:, 1:] - t_vals[:, :-1]
    last_delta = torch.full_like(delta_vals[:, :1], 1e10)
    delta_vals = torch.cat([delta_vals, last_delta], dim=-1)

    alpha_vals = 1.0 - torch.exp(-sigma_samples * delta_vals)

    trans_vals = torch.cumprod(
        torch.cat(
            [torch.ones((alpha_vals.shape[0], 1), device=alpha_vals.device), 1.0 - alpha_vals + 1e-10],
            dim=-1
        ),
        dim=-1
    )[:, :-1]

    weights = trans_vals * alpha_vals
    rgb_rendered = torch.sum(weights[..., None] * rgb_samples, dim=1)

    return rgb_rendered, weights


def predict_radiance(model, ray_origins, ray_directions, near=2.0, far=6.0, n_samples=32, perturb=True):
    sample_xyz, t_vals = sample_along_rays(
        ray_origins, ray_directions, near=near, far=far, n_samples=n_samples, perturb=perturb
    )

    expanded_dirs = ray_directions[:, None, :].expand_as(sample_xyz)

    num_rays, num_samples, _ = sample_xyz.shape
    flat_xyz = sample_xyz.reshape(-1, 3)
    flat_dirs = expanded_dirs.reshape(-1, 3)

    flat_rgb, flat_sigma = model(flat_xyz, flat_dirs)

    rgb_samples = flat_rgb.reshape(num_rays, num_samples, 3)
    sigma_samples = flat_sigma.reshape(num_rays, num_samples, 1)

    rendered_rgb, weights = volume_render(rgb_samples, sigma_samples, t_vals)
    return rendered_rgb, sample_xyz, weights


def volume_render_with_depth(rgb_samples, sigma_samples, t_vals):
    sigma_samples = sigma_samples.squeeze(-1)

    delta_vals = t_vals[:, 1:] - t_vals[:, :-1]
    last_delta = torch.full_like(delta_vals[:, :1], 1e10)
    delta_vals = torch.cat([delta_vals, last_delta], dim=-1)

    alpha_vals = 1.0 - torch.exp(-sigma_samples * delta_vals)

    trans_vals = torch.cumprod(
        torch.cat(
            [torch.ones((alpha_vals.shape[0], 1), device=alpha_vals.device), 1.0 - alpha_vals + 1e-10],
            dim=-1
        ),
        dim=-1
    )[:, :-1]

    weights = trans_vals * alpha_vals
    rgb_rendered = torch.sum(weights[..., None] * rgb_samples, dim=1)
    depth_rendered = torch.sum(weights * t_vals, dim=1)

    return rgb_rendered, depth_rendered, weights


def predict_radiance_and_depth(model, ray_origins, ray_directions, near=2.0, far=6.0, n_samples=32, perturb=True):
    sample_xyz, t_vals = sample_along_rays(
        ray_origins, ray_directions, near=near, far=far, n_samples=n_samples, perturb=perturb
    )

    expanded_dirs = ray_directions[:, None, :].expand_as(sample_xyz)

    num_rays, num_samples, _ = sample_xyz.shape
    flat_xyz = sample_xyz.reshape(-1, 3)
    flat_dirs = expanded_dirs.reshape(-1, 3)

    flat_rgb, flat_sigma = model(flat_xyz, flat_dirs)
    rgb_samples = flat_rgb.reshape(num_rays, num_samples, 3)
    sigma_samples = flat_sigma.reshape(num_rays, num_samples, 1)

    rendered_rgb, rendered_depth, weights = volume_render_with_depth(rgb_samples, sigma_samples, t_vals)
    return rendered_rgb, rendered_depth, sample_xyz, weights


@torch.no_grad()
def render_image(model, c2w, focal, image_height, image_width, device, near=2.0, far=6.0, n_samples=32, chunk_size=4096):
    yy, xx = torch.meshgrid(
        torch.arange(image_height, device=device),
        torch.arange(image_width, device=device),
        indexing="ij"
    )

    uv_grid = torch.stack([xx, yy], dim=-1).float() + 0.5
    uv_flat = uv_grid.reshape(-1, 2)

    ray_o, ray_d = pixels_to_rays(
        uv_coords=uv_flat,
        c2w=c2w,
        focal=focal,
        image_width=image_width,
        image_height=image_height
    )

    rgb_chunks = []

    for start_id in range(0, uv_flat.shape[0], chunk_size):
        sub_o = ray_o[start_id:start_id + chunk_size]
        sub_d = ray_d[start_id:start_id + chunk_size]

        rgb_chunk, _, _ = predict_radiance(
            model=model,
            ray_origins=sub_o,
            ray_directions=sub_d,
            near=near,
            far=far,
            n_samples=n_samples,
            perturb=False
        )
        rgb_chunks.append(rgb_chunk)

    rgb_full = torch.cat(rgb_chunks, dim=0)
    rgb_img = rgb_full.reshape(image_height, image_width, 3)
    return rgb_img


@torch.no_grad()
def render_depth_image(model, c2w, focal, image_height, image_width, device, near=2.0, far=6.0, n_samples=32, chunk_size=4096):
    yy, xx = torch.meshgrid(
        torch.arange(image_height, device=device),
        torch.arange(image_width, device=device),
        indexing="ij"
    )

    uv_grid = torch.stack([xx, yy], dim=-1).float() + 0.5
    uv_flat = uv_grid.reshape(-1, 2)

    ray_o, ray_d = pixels_to_rays(
        uv_coords=uv_flat,
        c2w=c2w,
        focal=focal,
        image_width=image_width,
        image_height=image_height
    )

    depth_chunks = []

    for start_id in range(0, uv_flat.shape[0], chunk_size):
        sub_o = ray_o[start_id:start_id + chunk_size]
        sub_d = ray_d[start_id:start_id + chunk_size]

        _, depth_chunk, _, _ = predict_radiance_and_depth(
            model=model,
            ray_origins=sub_o,
            ray_directions=sub_d,
            near=near,
            far=far,
            n_samples=n_samples,
            perturb=False
        )
        depth_chunks.append(depth_chunk)

    depth_full = torch.cat(depth_chunks, dim=0)
    depth_img = depth_full.reshape(image_height, image_width)
    return depth_img