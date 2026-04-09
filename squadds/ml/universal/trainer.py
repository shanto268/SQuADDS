"""Trainer for the Universal GNN (HeteroData version)."""

import os

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from squadds.ml.universal.model.gat_model import UniversalGNN
from squadds.ml.universal.model.loss import MaskedMultiTaskLoss


class UniversalTrainer:
    """Handles the training loop for the Heterogeneous Universal GNN."""

    def __init__(
        self,
        model: UniversalGNN,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = MaskedMultiTaskLoss()
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one forward pass and compute loss."""
        out = self.model(batch)

        node_preds = out["node_preds"]
        edge_preds = out["edge_preds"]

        node_targets = batch["component"].y
        edge_targets = batch["component", "physical", "component"].y

        return self.criterion(node_preds, node_targets, edge_preds, edge_targets)

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss_accum = 0.0
        node_loss_accum = 0.0
        edge_loss_accum = 0.0
        batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            loss, node_loss, edge_loss = self._step(batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss_accum += loss.item()
            node_loss_accum += node_loss.item()
            edge_loss_accum += edge_loss.item()
            batches += 1

        if batches == 0:
            return 0.0, 0.0, 0.0
        return total_loss_accum / batches, node_loss_accum / batches, edge_loss_accum / batches

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float, float]:
        """Evaluate on a validation/test set."""
        self.model.eval()
        total_loss_accum = 0.0
        node_loss_accum = 0.0
        edge_loss_accum = 0.0
        batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                batch = batch.to(self.device)
                loss, node_loss, edge_loss = self._step(batch)

                total_loss_accum += loss.item()
                node_loss_accum += node_loss.item()
                edge_loss_accum += edge_loss.item()
                batches += 1

        if batches == 0:
            return 0.0, 0.0, 0.0
        return total_loss_accum / batches, node_loss_accum / batches, edge_loss_accum / batches

    def train_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 10,
    ) -> dict:
        """Execute the full training loop with early stopping."""
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_node": [],
            "val_node": [],
            "train_edge": [],
            "val_edge": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")

            t_tot, t_node, t_edge = self.train_epoch(train_loader)
            v_tot, v_node, v_edge = self.evaluate(val_loader)

            history["train_loss"].append(t_tot)
            history["val_loss"].append(v_tot)
            history["train_node"].append(t_node)
            history["train_edge"].append(t_edge)
            history["val_node"].append(v_node)
            history["val_edge"].append(v_edge)

            print(f"  Train: Tot={t_tot:.4f} Node={t_node:.4f} Edge={t_edge:.4f}")
            print(f"  Val  : Tot={v_tot:.4f} Node={v_node:.4f} Edge={v_edge:.4f}")

            if v_tot < best_val_loss:
                best_val_loss = v_tot
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
                print("  [*] Model improved - checkpoint saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
