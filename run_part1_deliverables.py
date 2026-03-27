from part1_2d import train_on_two_images_for_progression, run_part1_comparison_grid

train_on_two_images_for_progression(
    image_paths=["lion_hw4.png", "car_hw4.png"],
    output_root="outputs_part1_two_images",
    pe_levels=10,
    hidden_width=256,
    lr=1e-2,
    num_iters=3000,
    batch_size=10000
)

run_part1_comparison_grid(
    image_path="lion_hw4.png",
    output_dir="outputs_part1_grid",
    pe_values=(4, 10),
    width_values=(64, 256),
    lr=1e-2,
    num_iters=2000,
    batch_size=10000
)