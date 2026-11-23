"""Fine-tune voxel models with different positive-voxel BCE weights."""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2

import dataset_location
from eval_model import evaluate
from model import SingleViewto3D
from r2n2_custom import R2N2


def model_args(device, batch_size):
    return argparse.Namespace(
        type="vox", device=device, load_feat=False, arch="resnet18",
        batch_size=batch_size, n_points=1000,
    )


def dataset(split, mesh):
    return R2N2(
        split, dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH, return_voxels=True, return_mesh=mesh,
    )


def train_variant(weight, cli, train_data, base_state):
    args = model_args(cli.device, cli.batch_size)
    model = SingleViewto3D(args).to(cli.device).train()
    model.load_state_dict(base_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=cli.lr)
    loader = torch.utils.data.DataLoader(
        train_data, batch_size=cli.batch_size, shuffle=True, drop_last=True,
        num_workers=cli.num_workers, persistent_workers=cli.num_workers > 0,
        pin_memory=True, collate_fn=collate_batched_R2N2,
    )
    iterator = iter(loader)
    positive_weight = torch.tensor(weight, device=cli.device)
    started = time.time()
    for step in range(cli.steps):
        try:
            feed = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            feed = next(iterator)
        images = feed["images"].squeeze(1).to(cli.device, non_blocking=True)
        targets = feed["voxels"].float().to(cli.device, non_blocking=True)
        predictions = model(images, args)
        if cli.loss_mode == "positive_weight":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                predictions, targets, pos_weight=positive_weight,
            )
        else:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)
            probabilities = predictions.sigmoid()
            intersection = (probabilities * targets).sum(dim=(1, 2, 3, 4))
            denominator = (probabilities + targets).sum(dim=(1, 2, 3, 4))
            dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
            loss = bce + weight * dice
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(
                f"weight={weight:g} step={step + 1}/{cli.steps} "
                f"loss={loss.item():.4f} time={time.time() - started:.0f}s",
                flush=True,
            )
    return model


@torch.no_grad()
def validate(model, cli, validation, count):
    # Use identical image views and surface samples for every loss setting.
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    args = model_args(cli.device, 1)
    model.eval()
    totals = {"precision": [], "recall": [], "f1": []}
    for index in range(min(count, len(validation))):
        feed = collate_batched_R2N2([validation[index]])
        images = feed["images"].squeeze(1).to(cli.device)
        predictions = model(images, args)
        metrics = evaluate(predictions, feed["mesh"], [0.05], args)
        totals["precision"].append(metrics["Precision@0.050000"].item())
        totals["recall"].append(metrics["Recall@0.050000"].item())
        totals["f1"].append(metrics["F1@0.050000"].item())
    return {key: float(np.mean(values)) for key, values in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0.5,1,2")
    parser.add_argument("--loss_mode", choices=("positive_weight", "dice"),
                        default="positive_weight")
    parser.add_argument("--steps", default=600, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--val_examples", default=100, type=int)
    parser.add_argument("--output_dir", default="report_results/voxel_loss")
    cli = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load("checkpoint_vox.pth", map_location="cpu")
    base_state = checkpoint["model_state_dict"]
    train_data = dataset("train", False)
    validation = dataset("test", True)
    results = {}
    best = None
    for weight in [float(value) for value in cli.weights.split(",")]:
        model = train_variant(weight, cli, train_data, base_state)
        metrics = validate(model, cli, validation, cli.val_examples)
        results[str(weight)] = metrics
        print(f"weight={weight:g} validation={metrics}", flush=True)
        if best is None or metrics["f1"] > best[0]:
            best = (metrics["f1"], weight)
        torch.save(
            {"step": checkpoint.get("step", 0) + cli.steps,
             "model_state_dict": model.state_dict(), "loss_mode": cli.loss_mode,
             "loss_weight": weight,
             "fine_tune_steps": cli.steps},
            output / f"checkpoint_pos_{weight:g}.pth",
        )
        del model
        torch.cuda.empty_cache()
    summary = {
        "fine_tune_steps": cli.steps, "learning_rate": cli.lr,
        "validation_examples": cli.val_examples, "results": results,
        "loss_mode": cli.loss_mode, "best_weight": best[1],
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    weights = [float(value) for value in results]
    f1 = [results[value]["f1"] for value in results]
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(weights, f1, marker="o")
    xlabel = "Positive-voxel BCE weight" if cli.loss_mode == "positive_weight" else "Dice-loss weight"
    axis.set(xlabel=xlabel, ylabel="F1@0.05", title="Effect of voxel loss weighting")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "loss_weight.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
