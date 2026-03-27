from train_nerf import train_nerf_main

train_nerf_main(
    npz_path="perrier_frames_10pct_200x200.npz",
    output_dir="outputs_part3_my_object",
    xyz_L=10,
    dir_L=4,
    hidden_dim=256,
    lr=5e-4,
    num_iters=1000,
    rays_per_batch=2048,
    near=0.02,
    far=0.5,
    n_samples=16
)