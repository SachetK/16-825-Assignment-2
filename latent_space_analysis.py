"""Analyze and visualize the voxel model's 512-D image latent space."""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform

import dataset_location
from r2n2_custom import R2N2
from render_eval import voxel_prediction_mesh
from render_report_gifs import colored_mesh, image_panel, load_model, render_mesh


@torch.no_grad()
def encode_dataset(model, data, device, batch_size):
    features, occupancy, images = [], [], []
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=4,
        persistent_workers=True, collate_fn=collate_batched_R2N2,
    )
    for feed in loader:
        batch = feed["images"].squeeze(1).to(device, non_blocking=True)
        normalized = model.normalize(batch.permute(0, 3, 1, 2))
        features.append(model.encoder(normalized).flatten(1).cpu())
        occupancy.append(feed["voxels"].float().flatten(1).mean(1))
        images.extend(batch.cpu())
    return torch.cat(features), torch.cat(occupancy), images


def save_scatter(scores, occupancy, explained, output):
    fig, axis = plt.subplots(figsize=(7.0, 5.2))
    points = axis.scatter(
        scores[:, 0], scores[:, 1], c=occupancy * 100,
        cmap="viridis", s=18, alpha=0.8, linewidths=0,
    )
    axis.set(
        xlabel=f"PC1 ({explained[0] * 100:.1f}% variance)",
        ylabel=f"PC2 ({explained[1] * 100:.1f}% variance)",
        title="PCA of 678 image embeddings",
    )
    fig.colorbar(points, ax=axis, label="Ground-truth occupied voxels (%)")
    axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_nearest_inputs(images, scores, targets, component, output):
    indices = [int(torch.argmin((scores[:, component] - target).abs())) for target in targets]
    panel_size = 180
    canvas = Image.new("RGB", (panel_size * len(indices), panel_size + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (index, target) in enumerate(zip(indices, targets)):
        panel = image_panel(images[index]).resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        canvas.paste(panel, (column * panel_size, 32))
        label = f"{target.item():+.1f} std"
        box = draw.textbbox((0, 0), label)
        draw.text((column * panel_size + (panel_size - box[2] + box[0]) / 2, 9), label, fill="black")
    canvas.save(output)
    return indices


@torch.no_grad()
def save_traversal(model, mean, direction, targets, scale, component, output, device):
    latent = torch.stack([mean + target * scale * direction for target in targets]).to(device)
    logits = model.decoder(latent.reshape(len(targets), 512, 1, 1, 1))
    meshes = [colored_mesh(voxel_prediction_mesh(logit), device) for logit in logits]
    panel_size = 180
    frames = []
    for azimuth in np.linspace(0, 360, 18, endpoint=False):
        rotation, translation = look_at_view_transform(
            dist=2.0, elev=20, azim=float(azimuth), device=device,
        )
        cameras = FoVPerspectiveCameras(device=device, R=rotation, T=translation)
        frame = Image.new("RGB", (panel_size * len(meshes), panel_size + 32), "white")
        draw = ImageDraw.Draw(frame)
        for column, (target, mesh) in enumerate(zip(targets, meshes)):
            rendered = Image.fromarray(render_mesh(mesh, cameras, device))
            rendered = rendered.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
            frame.paste(rendered, (column * panel_size, 32))
            label = f"PC{component + 1}: {target.item():+.1f} std"
            box = draw.textbbox((0, 0), label)
            draw.text((column * panel_size + (panel_size - box[2] + box[0]) / 2, 9), label, fill="black")
        frames.append(frame)
    frames[0].save(
        output, save_all=True, append_images=frames[1:],
        duration=110, loop=0, optimize=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoint_vox.pth")
    parser.add_argument("--output", default="proj2/data")
    parser.add_argument("--batch_size", default=32, type=int)
    cli = parser.parse_args()
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)

    data = R2N2(
        "test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH, return_voxels=True, return_mesh=False,
    )
    model, _ = load_model("vox", cli.checkpoint, cli.device)
    features, occupancy, images = encode_dataset(model, data, cli.device, cli.batch_size)
    mean = features.mean(0)
    centered = features - mean
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    scores = centered @ vh[:2].T
    explained = singular.square() / singular.square().sum()
    correlations = [float(torch.corrcoef(torch.stack((scores[:, i], occupancy)))[0, 1]) for i in range(2)]
    save_scatter(scores, occupancy, explained, output / "latent-pca.png")

    targets = torch.tensor((-1.5, -0.75, 0.0, 0.75, 1.5))
    nearest = {}
    for component in range(2):
        scale = scores[:, component].std()
        nearest[f"pc{component + 1}"] = save_nearest_inputs(
            images, scores / scores.std(0), targets, component,
            output / f"latent-pc{component + 1}-inputs.png",
        )
        save_traversal(
            model, mean, vh[component], targets, scale, component,
            output / f"latent-pc{component + 1}-traversal.gif", cli.device,
        )
    summary = {
        "num_examples": len(data),
        "explained_variance_percent": [float(value * 100) for value in explained[:2]],
        "occupancy_correlation": correlations,
        "nearest_example_indices": nearest,
    }
    (output / "latent-analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
