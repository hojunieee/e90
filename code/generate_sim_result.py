import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Setup ===
save_dir = "../E90/data/tag_capture_data/"
img_dir = os.path.join(save_dir, "images")
csv_path = os.path.join(save_dir, "tag_positions.csv")
output_video = os.path.join(save_dir, "quad_view_output.mp4")

# === Load CSV ===
df = pd.read_csv(csv_path)
grouped = df.groupby('Frame')
frame_ids = sorted(grouped.groups.keys())

# === Constants ===
fw, fh = 1920, 1080
canvas_size = (fh * 2, fw * 2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 5, (canvas_size[1], canvas_size[0]))

for frame_id in frame_ids:
    imgs = []
    for cam in range(3):
        path = os.path.join(img_dir, f"frame_{frame_id:05d}_cam{cam}.png")
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((fh, fw, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (fw, fh))
        imgs.append(img)

    # === Plot: X-Y positions per TagID ===
    fig, ax = plt.subplots(figsize=(fw / 100, fh / 100), dpi=100)
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.set_title(f"Frame {frame_id}")
    ax.grid(True)

    if frame_id in grouped.groups:
        frame_data = grouped.get_group(frame_id)
        for _, row in frame_data.iterrows():
            x, y, tag_id = row['X'], row['Y'], int(row['TagID'])
            ax.add_patch(plt.Circle((x, y), 0.1, color='black'))
            ax.text(x, y + 0.15, f"ID {tag_id}", ha='center', fontsize=8, color='blue')

    # Save and load the plot
    plot_path = os.path.join(save_dir, "plot_temp.png")
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    plot_img = cv2.imread(plot_path)
    plot_img = cv2.resize(plot_img, (fw, fh))

    # === Create 2x2 canvas ===
    top_row = np.hstack((imgs[0], imgs[1]))
    bottom_row = np.hstack((imgs[2], plot_img))
    final_canvas = np.vstack((top_row, bottom_row))

    out.write(final_canvas)

out.release()
print(f"[Done] Video saved to: {output_video}")
