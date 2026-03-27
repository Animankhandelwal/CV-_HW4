import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def ensure_dir(folder_path: str):
    os.makedirs(folder_path, exist_ok=True)


def save_plot_curve(y_values, save_path, title, xlabel="Iteration", ylabel="Value", x_values=None):
    plt.figure(figsize=(8, 5))
    if x_values is None:
        plt.plot(y_values)
    else:
        plt.plot(x_values, y_values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_image_grid(image_list, title_list, save_path, rows=2, cols=2):
    plt.figure(figsize=(10, 10))
    for idx, (img, title_text) in enumerate(zip(image_list, title_list)):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.title(title_text)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_gif_from_folder(frames_folder, gif_path, fps=12):
    image_files = sorted([
        os.path.join(frames_folder, x)
        for x in os.listdir(frames_folder)
        if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")
    ])

    frames = [imageio.imread(img_path) for img_path in image_files]
    imageio.mimsave(gif_path, frames, fps=fps)


def normalize_depth_for_display(depth_map):
    depth_map = np.asarray(depth_map)
    dmin = np.percentile(depth_map, 2)
    dmax = np.percentile(depth_map, 98)
    depth_norm = np.clip((depth_map - dmin) / (dmax - dmin + 1e-8), 0.0, 1.0)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    return depth_uint8


def visualize_rays_and_points(camera_origins, ray_origins, ray_directions, sampled_points, save_path, max_rays=100):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        camera_origins[:, 0],
        camera_origins[:, 1],
        camera_origins[:, 2],
        s=40,
        label="Cameras"
    )

    num_show = min(max_rays, ray_origins.shape[0])
    chosen_idx = np.linspace(0, ray_origins.shape[0] - 1, num_show).astype(int)

    for idx in chosen_idx:
        ro = ray_origins[idx]
        rd = ray_directions[idx]
        pts = sampled_points[idx]

        far_point = ro + 1.5 * rd
        ax.plot(
            [ro[0], far_point[0]],
            [ro[1], far_point[1]],
            [ro[2], far_point[2]],
            linewidth=0.8
        )

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)

    ax.set_title("Ray and Sample Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()