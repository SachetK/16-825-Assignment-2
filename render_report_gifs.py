"""Create rotating input/prediction/ground-truth GIFs for the HTML report."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch3d.renderer import (
    AlphaCompositor, FoVPerspectiveCameras, HardPhongShader, MeshRasterizer,
    MeshRenderer, PointLights, PointsRasterizationSettings, PointsRasterizer,
    PointsRenderer, RasterizationSettings, TexturesVertex, look_at_view_transform,
)
from pytorch3d.structures import Meshes, Pointclouds

import dataset_location
from model import SingleViewto3D
from r2n2_custom import R2N2
from render_eval import voxel_prediction_mesh


SIZE = 256
FRAMES = 18


def model_args(kind, device):
    return argparse.Namespace(
        type=kind, device=device, load_feat=False, arch="resnet18",
        batch_size=1, n_points=1000,
    )


def load_model(kind, checkpoint, device):
    args = model_args(kind, device)
    model = SingleViewto3D(args).to(device).eval()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model, args


def colored_mesh(mesh, device):
    mesh = mesh.to(device)
    verts = mesh.verts_list()[0].detach()
    color = verts.new_tensor((0.25, 0.55, 0.80)).expand_as(verts)
    return Meshes(
        verts=[verts], faces=[mesh.faces_list()[0]],
        textures=TexturesVertex(verts_features=[color]),
    )


def render_mesh(mesh, cameras, device):
    if mesh is None:
        return np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=SIZE, faces_per_pixel=1, bin_size=0,
            ),
        ),
        shader=HardPhongShader(
            device=device, cameras=cameras,
            lights=PointLights(device=device, location=((0.0, 2.0, -3.0),)),
        ),
    )
    image = renderer(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy()
    return (image * 255).astype(np.uint8)


def render_points(points, cameras):
    colors = points.new_tensor((0.20, 0.50, 0.85)).expand_as(points)
    cloud = Pointclouds(points=points, features=colors)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=cameras,
            raster_settings=PointsRasterizationSettings(
                image_size=SIZE, radius=0.012, points_per_pixel=10, bin_size=0,
            ),
        ),
        compositor=AlphaCompositor(background_color=(1.0, 1.0, 1.0)),
    )
    image = renderer(cloud)[0, ..., :3].clamp(0, 1).cpu().numpy()
    return (image * 255).astype(np.uint8)


def image_panel(image):
    array = image.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.clip(array * 255, 0, 255).astype(np.uint8)
    source = Image.fromarray(array).convert("RGB")
    source.thumbnail((SIZE, SIZE), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (SIZE, SIZE), "white")
    panel.paste(source, ((SIZE - source.width) // 2, (SIZE - source.height) // 2))
    return panel


def compose(input_image, predicted, target):
    labels = ("Input image", "Prediction", "Ground truth")
    panels = (input_image, Image.fromarray(predicted), Image.fromarray(target))
    frame = Image.new("RGB", (SIZE * 3, SIZE + 28), "white")
    draw = ImageDraw.Draw(frame)
    for index, (label, panel) in enumerate(zip(labels, panels)):
        x = index * SIZE
        frame.paste(panel, (x, 28))
        box = draw.textbbox((0, 0), label)
        width = box[2] - box[0]
        draw.text((x + (SIZE - width) / 2, 7), label, fill="black")
    return frame


@torch.no_grad()
def save_gif(image, prediction, target_mesh, kind, output, device):
    input_image = image_panel(image)
    target = colored_mesh(target_mesh, device)
    if kind == "vox":
        prediction = colored_mesh(voxel_prediction_mesh(prediction[0]), device)
    elif kind == "mesh":
        prediction = colored_mesh(prediction, device)
    else:
        prediction = prediction.detach()
    frames = []
    for azimuth in np.linspace(0, 360, FRAMES, endpoint=False):
        rotation, translation = look_at_view_transform(
            dist=2.0, elev=20, azim=float(azimuth), device=device,
        )
        cameras = FoVPerspectiveCameras(device=device, R=rotation, T=translation)
        if kind == "point":
            predicted_image = render_points(prediction, cameras)
        else:
            predicted_image = render_mesh(prediction, cameras, device)
        target_image = render_mesh(target, cameras, device)
        frames.append(compose(input_image, predicted_image, target_image))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=110, loop=0,
        optimize=True,
    )
    print(f"Saved {output}")


@torch.no_grad()
def predict_sample(model, args, sample):
    image = sample["images"]
    if image.ndim == 4:
        image = image[0]
    batched = image.unsqueeze(0).to(args.device)
    prediction = model(batched, args)
    target = Meshes(verts=[sample["verts"]], faces=[sample["faces"]])
    return image, prediction, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="proj2/data")
    parser.add_argument("--types", default="vox,point,mesh")
    parser.add_argument("--voxel_checkpoint", default="checkpoint_vox.pth")
    parser.add_argument("--skip_main", action="store_true")
    parser.add_argument("--skip_dataset_comparison", action="store_true")
    cli = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    output = Path(cli.output)
    dataset = R2N2(
        "test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH, return_voxels=True,
    )
    for kind in (() if cli.skip_main else cli.types.split(",")):
        checkpoint = cli.voxel_checkpoint if kind == "vox" else f"checkpoint_{kind}.pth"
        model, args = load_model(kind, checkpoint, cli.device)
        for number, index in enumerate((0, 300, 600), start=1):
            image, prediction, target = predict_sample(model, args, dataset[index])
            save_gif(image, prediction, target, kind, output / f"{kind}-{number}.gif", cli.device)
        del model
        torch.cuda.empty_cache()

    if cli.skip_dataset_comparison:
        return

    root = "r2n2_shapenet_dataset_full"
    full = R2N2(
        "test", f"{root}/shapenet", f"{root}/r2n2", f"{root}/split_3c.json",
        return_voxels=True,
    )
    chair_start = full.synset_start_idxs["03001627"]
    samples = [full[chair_start + number] for number in range(3)]
    for name, checkpoint in (
        ("chair-only", "checkpoint_vox_original.pth"),
        ("three-class", "full_dataset/checkpoint_vox.pth"),
    ):
        model, args = load_model("vox", checkpoint, cli.device)
        for number, sample in enumerate(samples):
            image, prediction, target = predict_sample(model, args, sample)
            save_gif(image, prediction, target, "vox", output / f"{name}-{number}.gif", cli.device)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
