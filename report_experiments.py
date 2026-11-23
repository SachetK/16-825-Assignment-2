"""Generate the hyperparameter and interpretation figures used by the report."""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes

import dataset_location
from eval_model import compute_sampling_metrics
from model import SingleViewto3D
from r2n2_custom import R2N2
from render_eval import voxel_prediction_mesh


def args_for_model(device):
    return argparse.Namespace(
        type="vox", device=device, load_feat=False, arch="resnet18",
        batch_size=1, n_points=1000,
    )


def load_model(device):
    args = args_for_model(device)
    model = SingleViewto3D(args).to(device).eval()
    checkpoint = torch.load("checkpoint_vox.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, args


@torch.no_grad()
def make_occlusion_figure(model, args, sample, output_dir):
    feed = collate_batched_R2N2([sample])
    image = feed["images"].squeeze(1).to(args.device)
    baseline = model(image, args).sigmoid()
    rows = cols = 7
    height, width = image.shape[1:3]
    variants = []
    for row in range(rows):
        for col in range(cols):
            occluded = image.clone()
            y0, y1 = row * height // rows, (row + 1) * height // rows
            x0, x1 = col * width // cols, (col + 1) * width // cols
            occluded[:, y0:y1, x0:x1] = 0
            variants.append(occluded[0])
    predictions = model(torch.stack(variants), args).sigmoid()
    sensitivity = (predictions - baseline).abs().mean(dim=(1, 2, 3, 4))
    sensitivity = sensitivity.reshape(rows, cols).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    rgb = image[0].cpu().numpy()
    axes[0].imshow(rgb)
    axes[0].set_title("Input image")
    axes[1].imshow(rgb)
    heat = axes[1].imshow(
        sensitivity, extent=(0, width, height, 0), interpolation="bicubic",
        cmap="magma", alpha=0.72,
    )
    axes[1].set_title("Voxel prediction sensitivity")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04, label="Mean occupancy change")
    fig.tight_layout()
    path = output_dir / "occlusion_vox.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


@torch.no_grad()
def threshold_study(model, args, dataset, output_dir, num_examples):
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    scores = {threshold: [] for threshold in thresholds}
    for index in range(min(num_examples, len(dataset))):
        feed = collate_batched_R2N2([dataset[index]])
        image = feed["images"].squeeze(1).to(args.device)
        logits = model(image, args)[0].cpu()
        mesh_gt = feed["mesh"]
        gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
        for threshold in thresholds:
            mesh = voxel_prediction_mesh(logits, threshold)
            if mesh is None:
                scores[threshold].append(0.0)
                continue
            pred_points = sample_points_from_meshes(mesh, args.n_points)
            metric = compute_sampling_metrics(pred_points, gt_points, [0.05])
            scores[threshold].append(metric["F1@0.050000"].item())
        if (index + 1) % 20 == 0:
            print(f"Threshold study: {index + 1}/{num_examples}")
    means = [float(np.mean(scores[t])) for t in thresholds]
    summary = {"num_examples": num_examples, "thresholds": thresholds, "f1_at_0.05": means}
    (output_dir / "voxel_threshold.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(thresholds, means, marker="o", color="#2563eb", linewidth=2)
    best = int(np.argmax(means))
    axis.scatter([thresholds[best]], [means[best]], color="#7c3aed", s=70, zorder=3)
    axis.set(xlabel="Occupancy threshold", ylabel="Mean F1@0.05",
             title=f"Voxel threshold sensitivity ({num_examples} test examples)")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path = output_dir / "voxel_threshold.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}: {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_examples", default=100, type=int)
    parser.add_argument("--output_dir", default="report_results")
    cli = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = R2N2(
        "test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH, return_voxels=True,
    )
    model, args = load_model(cli.device)
    make_occlusion_figure(model, args, dataset[0], output_dir)
    threshold_study(model, args, dataset, output_dir, cli.num_examples)


if __name__ == "__main__":
    main()
