"""Render fitted and target voxel surfaces with the same PyTorch3D camera."""

import mcubes
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes


@torch.no_grad()
def render_voxel_comparison(voxels_src, voxels_tgt, output_path="voxel_comparison.png"):
    """Source is logits; target is binary. Inputs have shape (1, Z, Y, X)."""
    device = voxels_src.device
    R, T = look_at_view_transform(dist=1.8, elev=20, azim=40, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
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
    for grid in (voxels_src.sigmoid(), voxels_tgt):
        # Marching cubes expects a scalar volume. Reorder storage (Z,Y,X)
        # into world (X,Y,Z), and pad to close surfaces at the grid boundary.
        volume = grid.detach().squeeze(0).permute(2, 1, 0).cpu().numpy()
        volume = (volume >= 0.5).astype(np.float32)
        if not volume.any():
            images.append(np.ones((512, 512, 3), dtype=np.float32))
            continue
        vertices, faces = mcubes.marching_cubes(np.pad(volume, 1), 0.5)
        # Undo padding and use the same grid-to-world transform for both
        # objects, matching utils_vox.Mem2Ref: index / resolution - 0.5.
        vertices = (vertices - 1) / np.array(volume.shape) - 0.5
        verts = torch.as_tensor(vertices, dtype=torch.float32, device=device)
        faces = torch.as_tensor(faces.astype(np.int64), device=device)
        color = verts.new_tensor((0.35, 0.65, 0.90)).expand_as(verts)
        mesh = Meshes(
            verts=[verts], faces=[faces],
            textures=TexturesVertex(verts_features=[color]),
        )
        images.append(renderer(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy())

    # Matplotlib only arranges the images; PyTorch3D does all 3D rendering.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, image, title in zip(axes, images, ("Optimized voxels", "Ground truth")):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved voxel comparison to {output_path}")
