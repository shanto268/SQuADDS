"""Trainer for the Universal GNN Model."""

import os

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from squadds.ml.universal.model.gat_model import UniversalGNN
from squadds.ml.universal.model.loss import MaskedMultiTaskLoss


class UniversalTrainer:
    """Handles the training loop for the Universal Graph Pipeline."""

    def __init__(
        self,
        model: UniversalGNN,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        node_weights: list[float] | None = None,
        edge_weights: list[float] | None = None,
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Args:
            model: The GATv2 model to train.
            learning_rate: Optimizer learning rate.
            device: 'cpu' or 'cuda'.
            node_weights: Optional scaling weights for node targets.
            edge_weights: Optional scaling weights for edge targets.
            checkpoint_dir: Directory to save model checkpoints.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = MaskedMultiTaskLoss(node_weights=node_weights, edge_weights=edge_weights)
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (avg_total_loss, avg_node_loss, avg_edge_loss).
        """
        self.model.train()
        total_loss_accum = 0.0
        node_loss_accum = 0.0
        edge_loss_accum = 0.0
        batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            node_preds, edge_preds = self.model(batch)

            loss, node_loss, edge_loss = self.criterion(node_preds, batch.y, edge_preds, batch.y_edge)

            loss.backward()

            # gradient clipping for stability
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
        """Evaluate the model on a validation/test set.

        Returns:
            Tuple of (avg_total_loss, avg_node_loss, avg_edge_loss).
        """
        self.model.eval()
        total_loss_accum = 0.0
        node_loss_accum = 0.0
        edge_loss_accum = 0.0
        batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                batch = batch.to(self.device)

                node_preds, edge_preds = self.model(batch)

                loss, node_loss, edge_loss = self.criterion(node_preds, batch.y, edge_preds, batch.y_edge)

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
        """Execute the full training loop with early stopping.

        Args:
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            epochs: Maximum number of epochs.
            patience: Early stopping patience.

        Returns:
            Dictionary containing history of losses.
        """
        history = {"train_loss": [], "val_loss": [], "train_node": [], "val_node": [], "train_edge": [], "val_edge": []}

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
