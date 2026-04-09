"""Trainer for the Universal GNN (node + edge targets)."""

import os

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from squadds.ml.universal.model.gat_model import UniversalGNN


class UniversalTrainer:
    """Training loop for the Heterogeneous Universal GNN.

    ALL nodes predict ALL node targets (qubit_freq, anharmonicity, cavity_freq).
    ALL physical edges predict ALL edge targets (g, kappa).
    The GNN learns correlations through message passing.
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

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one forward pass and compute loss."""
        out = self.model(batch)

        # Node loss
        node_preds = out["node_preds"]
        node_targets = batch["component"].y
        mask_n = ~torch.isnan(node_targets)
        if mask_n.any():
            loss_node = self.criterion(node_preds[mask_n], node_targets[mask_n])
        else:
            loss_node = (0.0 * node_preds).sum()

        # Edge loss
        edge_preds = out["edge_preds"]
        phys_key = ("component", "physical", "component")
        if phys_key in batch.edge_types and hasattr(batch[phys_key], "y"):
            edge_targets = batch[phys_key].y
            mask_e = ~torch.isnan(edge_targets)
            if mask_e.any():
                loss_edge = self.criterion(edge_preds[mask_e], edge_targets[mask_e])
            else:
                loss_edge = (0.0 * edge_preds).sum()
        else:
            loss_edge = torch.tensor(0.0, device=self.device)

        total_loss = loss_node + loss_edge
        return total_loss, loss_node, loss_edge

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        tot_acc, node_acc, edge_acc = 0.0, 0.0, 0.0
        batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            total, node, edge = self._step(batch)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            tot_acc += total.item()
            node_acc += node.item()
            edge_acc += edge.item()
            batches += 1

        n = max(batches, 1)
        return tot_acc / n, node_acc / n, edge_acc / n

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        tot_acc, node_acc, edge_acc = 0.0, 0.0, 0.0
        batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                batch = batch.to(self.device)
                total, node, edge = self._step(batch)
                tot_acc += total.item()
                node_acc += node.item()
                edge_acc += edge.item()
                batches += 1

        n = max(batches, 1)
        return tot_acc / n, node_acc / n, edge_acc / n

    def train_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 10,
    ) -> dict:
        """Full training loop with early stopping."""
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
            t_tot, t_node, t_edge = self.train_epoch(train_loader)
            v_tot, v_node, v_edge = self.evaluate(val_loader)

            history["train_loss"].append(t_tot)
            history["val_loss"].append(v_tot)
            history["train_node"].append(t_node)
            history["val_node"].append(v_node)
            history["train_edge"].append(t_edge)
            history["val_edge"].append(v_edge)

            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"Train={t_tot:.6f} (N={t_node:.6f} E={t_edge:.6f})  "
                f"Val={v_tot:.6f} (N={v_node:.6f} E={v_edge:.6f})",
                end="",
            )

            if v_tot < best_val_loss:
                best_val_loss = v_tot
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
                print("  [*]", end="")
            else:
                patience_counter += 1

            print()

            if patience_counter >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

        return history

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
