"""Compare chair reconstruction after one-class and three-class voxel training."""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2

from eval_model import evaluate
from model import SingleViewto3D
from r2n2_custom import R2N2
from render_eval import render_prediction_comparison


def model_args(device):
    return argparse.Namespace(
        type="vox", device=device, load_feat=False, arch="resnet18",
        batch_size=1, n_points=1000,
    )


def load_model(path, args):
    model = SingleViewto3D(args).to(args.device).eval()
    checkpoint = torch.load(path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_examples", default=100, type=int)
    parser.add_argument("--output_dir", default="report_results/dataset_comparison")
    parser.add_argument("--chair_checkpoint", default="checkpoint_vox_original.pth")
    cli = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    args = model_args(cli.device)
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = "r2n2_shapenet_dataset_full"
    dataset = R2N2(
        "test", f"{root}/shapenet", f"{root}/r2n2", f"{root}/split_3c.json",
        return_voxels=True,
    )
    chair_start = dataset.synset_start_idxs["03001627"]
    models = {
        "Chair only": load_model(cli.chair_checkpoint, args),
        "Three classes": load_model("full_dataset/checkpoint_vox.pth", args),
    }
    values = {name: {"precision": [], "recall": [], "f1": []} for name in models}
    for offset in range(cli.num_examples):
        feed = collate_batched_R2N2([dataset[chair_start + offset]])
        image = feed["images"].squeeze(1).to(args.device)
        for name, model in models.items():
            # Use identical surface samples for the paired model comparison.
            torch.manual_seed(offset)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(offset)
            prediction = model(image, args)
            metrics = evaluate(prediction, feed["mesh"], [0.05], args)
            values[name]["precision"].append(metrics["Precision@0.050000"].item())
            values[name]["recall"].append(metrics["Recall@0.050000"].item())
            values[name]["f1"].append(metrics["F1@0.050000"].item())
            if offset < 3:
                render_prediction_comparison(
                    feed["images"], prediction, feed["mesh"], "vox",
                    output / name.lower().replace(" ", "_"), offset, metrics,
                )
        if (offset + 1) % 20 == 0:
            print(f"Dataset comparison: {offset + 1}/{cli.num_examples}")
    summary = {
        name: {metric: float(np.mean(samples)) for metric, samples in metrics.items()}
        for name, metrics in values.items()
    }
    summary["num_chair_examples"] = cli.num_examples
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    labels = list(models)
    metric_names = ("precision", "recall", "f1")
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(metric_names))
    width = 0.36
    for index, (label, color) in enumerate(zip(labels, ("#2563eb", "#7c3aed"))):
        values = [summary[label][metric] for metric in metric_names]
        bars = axis.bar(x + (index - 0.5) * width, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axis.set_xticks(x, ("Precision", "Recall", "F1"))
    axis.set(ylabel="Mean score at 0.05", title=f"Training-set comparison ({cli.num_examples} matched chairs)")
    axis.set_ylim(0, 72)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
