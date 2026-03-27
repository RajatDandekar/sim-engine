"""Training pipeline — run with: python -m sim_engine train"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import SimDataset


def train_mlp(data_dir, epochs=50, batch_size=64, lr=1e-4, output_dir=None, device="cpu"):
    """Train the MLP policy via behavior cloning."""
    from .policies.mlp_policy import MLPPolicy

    output_dir = output_dir or os.path.join(config.DEFAULT_OUTPUT_DIR, "mlp")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = SimDataset(data_dir=data_dir, load_images=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    stats = dataset.get_normalization_stats()

    # Create policy
    policy = MLPPolicy(device=device)
    policy.set_normalization(stats)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    print(f"Training MLP on {len(dataset)} samples, {epochs} epochs")
    print(f"Device: {device}, Batch size: {batch_size}, LR: {lr}")
    print(f"Output: {output_dir}")

    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            states = batch["state"].float().to(policy.device)
            actions = batch["action"].float().to(policy.device)

            pred_actions = policy.model(states)
            loss = loss_fn(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            policy.save(os.path.join(output_dir, "best.pt"))

    policy.save(os.path.join(output_dir, "final.pt"))
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Saved: {output_dir}/best.pt, {output_dir}/final.pt")


def main():
    parser = argparse.ArgumentParser(description="Train a policy on recorded demos")
    parser.add_argument("--policy", choices=["mlp", "act", "diffusion", "smolvla"], default="mlp")
    parser.add_argument("--data_dir", type=str, default=config.DEFAULT_DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.policy == "mlp":
        train_mlp(args.data_dir, args.epochs, args.batch_size, args.lr, args.output_dir, args.device)
    else:
        print(f"Training for '{args.policy}' — implement this in policies/{args.policy}_policy.py!")
        print("See policies/mlp_policy.py for a complete example.")


if __name__ == "__main__":
    main()
