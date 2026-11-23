"""Visualize the shape manifold learned by the voxel model."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform

import dataset_location
from r2n2_custom import R2N2
from render_eval import voxel_prediction_mesh
from render_report_gifs import colored_mesh, image_panel, load_model, render_mesh


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoint_vox.pth")
    parser.add_argument("--output", default="proj2/data")
    parser.add_argument("--indices", default="0,600")
    cli = parser.parse_args()
    random.seed(7)
    torch.manual_seed(7)

    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    data = R2N2(
        "test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH, return_voxels=True,
    )
    model, args = load_model("vox", cli.checkpoint, cli.device)
    indices = [int(value) for value in cli.indices.split(",")]
    images = []
    for index in indices:
        image = data[index]["images"]
        if image.ndim == 4:
            image = image[0]
        images.append(image)

    batch = torch.stack(images).to(cli.device)
    normalized = model.normalize(batch.permute(0, 3, 1, 2))
    features = model.encoder(normalized).flatten(1)
    alphas = torch.linspace(0, 1, 5, device=cli.device)
    mixed = torch.stack([(1 - alpha) * features[0] + alpha * features[1] for alpha in alphas])
    logits = model.decoder(mixed.reshape(5, 512, 1, 1, 1))
    meshes = [colored_mesh(voxel_prediction_mesh(logit), cli.device) for logit in logits]

    source = Image.new("RGB", (512, 284), "white")
    draw = ImageDraw.Draw(source)
    for column, (label, image) in enumerate(zip(("Chair A", "Chair B"), images)):
        panel = image_panel(image)
        source.paste(panel, (column * 256, 28))
        box = draw.textbbox((0, 0), label)
        draw.text((column * 256 + (256 - (box[2] - box[0])) / 2, 7), label, fill="black")
    source.save(output / "latent-inputs.png")

    frames = []
    panel_size = 190
    for azimuth in np.linspace(0, 360, 18, endpoint=False):
        rotation, translation = look_at_view_transform(
            dist=2.0, elev=20, azim=float(azimuth), device=cli.device,
        )
        cameras = FoVPerspectiveCameras(device=cli.device, R=rotation, T=translation)
        frame = Image.new("RGB", (panel_size * 5, panel_size + 30), "white")
        draw = ImageDraw.Draw(frame)
        for column, (alpha, mesh) in enumerate(zip(alphas, meshes)):
            rendered = Image.fromarray(render_mesh(mesh, cameras, cli.device))
            rendered = rendered.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
            frame.paste(rendered, (column * panel_size, 30))
            label = f"alpha = {alpha.item():.2f}"
            box = draw.textbbox((0, 0), label)
            draw.text(
                (column * panel_size + (panel_size - (box[2] - box[0])) / 2, 8),
                label, fill="black",
            )
        frames.append(frame)
    frames[0].save(
        output / "latent-interpolation.gif", save_all=True,
        append_images=frames[1:], duration=110, loop=0, optimize=True,
    )
    print(f"Saved {output / 'latent-inputs.png'}")
    print(f"Saved {output / 'latent-interpolation.gif'}")


if __name__ == "__main__":
    main()
