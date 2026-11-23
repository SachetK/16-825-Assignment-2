"""PyTorch3D renders for image-to-3D evaluation comparisons."""
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch3d.renderer import (
    AlphaCompositor, HardPhongShader, MeshRasterizer, MeshRenderer,
    PointLights, PointsRasterizationSettings, PointsRasterizer, PointsRenderer,
    RasterizationSettings, TexturesVertex,
)
from pytorch3d.structures import Meshes, Pointclouds

from render_fit import _cameras


@torch.no_grad()
def voxel_prediction_mesh(logits, threshold=0.5):
    """Convert one (1, Z, Y, X) logit grid to the dataset's mesh coordinates."""
    import mcubes

    volume = (logits.detach().squeeze(0).sigmoid() >= threshold).cpu().numpy()
    if not volume.any():
        return None
    # Match the assignment evaluator's R2N2 convention: marching-cubes axes
    # followed by a half-turn about Y. Padding closes surfaces at boundaries.
    volume = volume.astype(np.float32)
    vertices, faces = mcubes.marching_cubes(np.pad(volume, 1), 0.5)
    vertices = (vertices - 1) / np.array(volume.shape) - 0.5
    vertices *= np.array((-1.0, 1.0, -1.0))
    return Meshes(
        verts=[torch.as_tensor(vertices, dtype=torch.float32, device=logits.device)],
        faces=[torch.as_tensor(faces.astype(np.int64), device=logits.device)],
    )


@torch.no_grad()
def render_prediction_comparison(
    images, predictions, mesh_gt, kind, output_dir, step, metrics=None
):
    """Save RGB / prediction / ground-truth panels for voxels, points, or meshes."""
    device = predictions.device
    cameras = _cameras(device)
    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            # Disable coarse bins so dense ground-truth meshes cannot overflow them.
            raster_settings=RasterizationSettings(image_size=512, faces_per_pixel=1, bin_size=0),
        ),
        shader=HardPhongShader(
            device=device, cameras=cameras,
            lights=PointLights(device=device, location=((0.0, 2.0, -3.0),)),
        ),
    )
    if kind == 'point':
        point_renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras,
                raster_settings=PointsRasterizationSettings(
                    image_size=512, radius=0.01, points_per_pixel=10, bin_size=0,
                ),
            ),
            compositor=AlphaCompositor(background_color=(1.0, 1.0, 1.0)),
        )

    def render_mesh(mesh):
        if mesh is None:
            return np.ones((512, 512, 3), dtype=np.float32)
        mesh = mesh.to(device)
        verts = mesh.verts_list()[0].detach()
        colors = verts.new_tensor((0.35, 0.65, 0.90)).expand_as(verts)
        colored = Meshes(
            verts=[verts], faces=mesh.faces_list(),
            textures=TexturesVertex(verts_features=[colors]),
        )
        return mesh_renderer(colored)[0, ..., :3].clamp(0, 1).cpu().numpy()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if images.ndim == 5:
        images = images[:, 0]
    batch_size = len(predictions) if kind == 'mesh' else predictions.shape[0]
    for index in range(batch_size):
        empty_prediction = False
        if kind == 'vox':
            predicted_mesh = voxel_prediction_mesh(predictions[index])
            empty_prediction = predicted_mesh is None
            predicted = render_mesh(predicted_mesh)
            prediction_title = 'Predicted voxels'
        elif kind == 'point':
            points = predictions[index:index + 1].detach()
            colors = points.new_tensor((0.35, 0.65, 0.90)).expand_as(points)
            predicted = point_renderer(Pointclouds(points=points, features=colors))
            predicted = predicted[0, ..., :3].clamp(0, 1).cpu().numpy()
            prediction_title = 'Predicted points'
        else:
            predicted = render_mesh(predictions[index])
            prediction_title = 'Predicted mesh'
        if metrics is not None:
            f1 = metrics['F1@0.050000'][index].item()
            precision = metrics['Precision@0.050000'][index].item()
            recall = metrics['Recall@0.050000'][index].item()
            prediction_title += f'\nF1@0.05 {f1:.1f}  P {precision:.1f}  R {recall:.1f}'
        panels = (images[index].detach().cpu().numpy(), predicted, render_mesh(mesh_gt[index]))
        titles = ('Input RGB', prediction_title, 'Ground-truth mesh')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
        for ax, panel, title in zip(axes, panels, titles):
            ax.imshow(panel)
            ax.set_title(title)
            ax.axis('off')
        if empty_prediction:
            max_probability = predictions[index].sigmoid().max().item()
            axes[1].text(
                0.5, 0.5,
                f'No occupied voxels at threshold 0.5\nMax occupancy probability: {max_probability:.4f}',
                ha='center', va='center', transform=axes[1].transAxes,
            )
            print(f'Empty voxel prediction at batch {step}, item {index}: max probability {max_probability:.4f}')
        path = Path(output_dir) / f'{step:06d}_{index:02d}_{kind}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f'Saved visualization to {path}')
