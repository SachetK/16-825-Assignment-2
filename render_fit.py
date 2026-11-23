"""Render fitted point clouds and meshes alongside their ground truth."""

import torch
from matplotlib import pyplot as plt
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes, Pointclouds


def _cameras(device):
    R, T = look_at_view_transform(dist=1.8, elev=20, azim=40, device=device)
    return FoVPerspectiveCameras(device=device, R=R, T=T)


def _save_comparison(images, title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, image, label in zip(axes, images, (title, "Ground truth")):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison to {output_path}")


@torch.no_grad()
def render_pointcloud_comparison(
    points_src, points_tgt, output_path="pointcloud_comparison.png"
):
    """Render fitted and target tensors of shape (1, N, 3) with one camera."""
    cameras = _cameras(points_src.device)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=cameras,
            raster_settings=PointsRasterizationSettings(
                image_size=512, radius=0.01, points_per_pixel=10,
            ),
        ),
        compositor=AlphaCompositor(background_color=(1.0, 1.0, 1.0)),
    )
    images = []
    for points in (points_src, points_tgt):
        points = points.detach()
        colors = points.new_tensor((0.35, 0.65, 0.90)).expand_as(points)
        cloud = Pointclouds(points=points, features=colors)
        images.append(renderer(cloud)[0, ..., :3].clamp(0, 1).cpu().numpy())
    _save_comparison(images, "Optimized point cloud", output_path)


@torch.no_grad()
def render_mesh_comparison(mesh_src, mesh_tgt, output_path="mesh_comparison.png"):
    """Render fitted and target single meshes without changing their textures."""
    device = mesh_src.device
    cameras = _cameras(device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=512, blur_radius=0.0, faces_per_pixel=1,
            ),
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=PointLights(device=device, location=((0.0, 2.0, -3.0),)),
        ),
    )
    images = []
    for mesh in (mesh_src, mesh_tgt):
        verts = mesh.verts_list()[0].detach()
        colors = verts.new_tensor((0.35, 0.65, 0.90)).expand_as(verts)
        colored_mesh = Meshes(
            verts=[verts], faces=[mesh.faces_list()[0].detach()],
            textures=TexturesVertex(verts_features=[colors]),
        )
        images.append(renderer(colored_mesh)[0, ..., :3].clamp(0, 1).cpu().numpy())
    _save_comparison(images, "Optimized mesh", output_path)
