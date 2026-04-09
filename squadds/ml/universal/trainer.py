"""Trainer for the Universal GNN (HeteroData, node-only targets)."""

import os

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from squadds.ml.universal.model.gat_model import UniversalGNN


class UniversalTrainer:
    """Training loop for the Heterogeneous Universal GNN.

    All component nodes predict all 5 Hamiltonian targets.
    Loss is simple MSE on all node predictions (no NaN masking
    during training on the standard topology).
    """

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
        self.criterion = torch.nn.MSELoss()
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def _step(self, batch) -> torch.Tensor:
        """Run one forward pass and compute loss."""
        out = self.model(batch)
        node_preds = out["node_preds"]
        node_targets = batch["component"].y

        # Standard MSE on ALL node predictions — model learns all targets
        # NaN masking only needed at inference on new topologies
        mask = ~torch.isnan(node_targets)
        if mask.any():
            loss = self.criterion(node_preds[mask], node_targets[mask])
        else:
            loss = (0.0 * node_preds).sum()
        return loss

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch. Returns avg loss."""
        self.model.train()
        total_loss = 0.0
        batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            loss = self._step(batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        return total_loss / max(batches, 1)

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set. Returns avg loss."""
        self.model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                batch = batch.to(self.device)
                loss = self._step(batch)
                total_loss += loss.item()
                batches += 1

        return total_loss / max(batches, 1)

    def train_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 10,
    ) -> dict:
        """Full training loop with early stopping."""
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t_loss = self.train_epoch(train_loader)
            v_loss = self.evaluate(val_loader)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)

            print(
                f"Epoch {epoch}/{epochs}  Train={t_loss:.6f}  Val={v_loss:.6f}",
                end="",
            )

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
                print("  [*] saved", end="")
            else:
                patience_counter += 1

            print()

            if patience_counter >= patience:
                print(f"Early stopping after {epoch} epochs.")
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
