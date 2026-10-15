import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from models.cross_modal_predictor import BiDirectionalInfoNCELoss, CrossModalPredictor
from utils.common import ensure_dir, resolve_device, set_seed
from .training_utils import CosineWarmupScheduler, save_checkpoint

logger = logging.getLogger(__name__)


class FeaturePairDataset(Dataset):
    """PyTorch Dataset yielding paired visual and textual representation vectors."""

    def __init__(self, visual_features: torch.Tensor, text_features: torch.Tensor, labels: List[Any]):
        self.visual = visual_features
        self.text = text_features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.visual)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        return self.visual[idx], self.text[idx], self.labels[idx]


class AlignmentTrainer:
    """
    Trains the VL-JEPA CrossModalPredictor on extracted video and text features.
    Uses Bi-directional InfoNCE loss to align V-JEPA 2 visual dynamics with text embeddings.
    """

    def __init__(self, config: Dict[str, Any], visual_dim: Optional[int] = None):
        self.config = config
        self.device = resolve_device(config.get("system", {}).get("device", "auto"))

        pcfg = config["models"]["predictor"]
        tcfg = config.get("training", {})

        self.epochs = int(tcfg.get("epochs", 25))
        self.batch_size = int(tcfg.get("batch_size", 64))
        self.lr = float(tcfg.get("learning_rate", 1.0e-4))
        self.weight_decay = float(tcfg.get("weight_decay", 0.01))
        self.temperature = float(pcfg.get("temperature", 0.07))

        self.save_dir = ensure_dir(tcfg.get("save_dir", "results/checkpoints"))

        v_dim = visual_dim or int(config["models"]["vljepa"].get("visual_dim", 768))
        self._init_model(v_dim)

    def _init_model(self, visual_dim: int) -> None:
        pcfg = self.config["models"]["predictor"]
        self.model = CrossModalPredictor(
            visual_dim=visual_dim,
            text_dim=int(self.config["models"]["vljepa"].get("text_dim", 768)),
            d_model=int(pcfg.get("d_model", 1536)),
            shared_dim=int(pcfg.get("shared_dim", 768)),
            num_layers=int(pcfg.get("num_layers", 4)),
            num_heads=int(pcfg.get("num_heads", 16)),
            dim_feedforward=int(pcfg.get("dim_feedforward", 4096)),
            dropout=float(pcfg.get("dropout", 0.1)),
        ).to(self.device)

        self.criterion = BiDirectionalInfoNCELoss(temperature=self.temperature)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_on_features(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: List[Any],
        val_split: float = 0.2,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end training and validation loop on feature tensors.
        """
        feat_dim = visual_features.shape[-1]
        if self.model.visual_dim != feat_dim:
            logger.info("Aligning CrossModalPredictor visual_dim (%d -> %d) with feature tensor", self.model.visual_dim, feat_dim)
            self._init_model(feat_dim)

        set_seed(self.config.get("system", {}).get("seed", 42))
        total_samples = len(visual_features)
        indices = np.random.permutation(total_samples)

        split_idx = int(total_samples * (1 - val_split))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        train_ds = FeaturePairDataset(visual_features[train_idx], text_features[train_idx], [labels[i] for i in train_idx])
        val_ds = FeaturePairDataset(visual_features[val_idx], text_features[val_idx], [labels[i] for i in val_idx])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        total_steps = self.epochs * len(train_loader)
        warmup_steps = int(self.config.get("training", {}).get("warmup_epochs", 3)) * len(train_loader)
        scheduler = CosineWarmupScheduler(self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_recall_at_1": [],
            "val_recall_at_5": [],
        }

        best_val_recall = -1.0
        best_checkpoint_path = str(self.save_dir / "vljepa_predictor_best.pt")
        dataset_checkpoint_path = str(self.save_dir / f"vljepa_predictor_{checkpoint_name}.pt") if checkpoint_name else None

        logger.info("Starting CrossModalPredictor training on %d samples (%d train, %d val, %d epochs)", total_samples, len(train_ds), len(val_ds), self.epochs)

        for epoch in range(1, self.epochs + 1):
            # --- Training Phase ---
            self.model.train()
            total_train_loss = 0.0
            for v_batch, t_batch, _ in train_loader:
                v_batch = v_batch.to(self.device)
                t_batch = t_batch.to(self.device)

                self.optimizer.zero_grad()
                pred_v, target_t = self.model(v_batch, t_batch)
                loss = self.criterion(pred_v, target_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_loader))

            # --- Validation Phase ---
            self.model.eval()
            total_val_loss = 0.0
            all_val_pred_v = []
            all_val_target_t = []
            all_val_labels = []

            with torch.no_grad():
                for v_batch, t_batch, l_batch in val_loader:
                    v_batch = v_batch.to(self.device)
                    t_batch = t_batch.to(self.device)

                    pred_v, target_t = self.model(v_batch, t_batch)
                    loss = self.criterion(pred_v, target_t)
                    total_val_loss += loss.item()

                    all_val_pred_v.append(pred_v.cpu())
                    all_val_target_t.append(target_t.cpu())
                    all_val_labels.extend(l_batch)

            avg_val_loss = total_val_loss / max(1, len(val_loader))

            # Compute Retrieval Recall@1 and Recall@5 on validation set
            if all_val_pred_v:
                val_v_matrix = torch.cat(all_val_pred_v, dim=0)  # [N_val, D]
                val_t_matrix = torch.cat(all_val_target_t, dim=0)  # [N_val, D]

                # Pairwise cosine similarity matrix
                sim_matrix = torch.matmul(val_t_matrix, val_v_matrix.t()).numpy()  # [N_val, N_val]
                recall_1, recall_5 = self._compute_recall_metrics(sim_matrix, all_val_labels)
            else:
                recall_1, recall_5 = 0.0, 0.0

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_recall_at_1"].append(recall_1)
            history["val_recall_at_5"].append(recall_5)

            logger.info(
                "Epoch [%d/%d] - Train Loss: %.4f | Val Loss: %.4f | Val R@1: %.3f | Val R@5: %.3f",
                epoch, self.epochs, avg_train_loss, avg_val_loss, recall_1, recall_5,
            )

            # Checkpoint best model based on validation recall
            if recall_5 >= best_val_recall:
                best_val_recall = recall_5
                save_checkpoint(self.model.state_dict(), epoch, avg_val_loss, recall_5, best_checkpoint_path)
                if dataset_checkpoint_path:
                    save_checkpoint(self.model.state_dict(), epoch, avg_val_loss, recall_5, dataset_checkpoint_path)

        # Save training history JSON
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        logger.info("Training complete. Best model saved to %s (Val R@5: %.3f)", best_checkpoint_path, best_val_recall)
        return history

    def _compute_recall_metrics(self, sim_matrix: np.ndarray, labels: List[Any]) -> Tuple[float, float]:
        """Calculates Recall@1 and Recall@5 from similarity matrix and ground-truth labels."""
        n = len(labels)
        if n == 0:
            return 0.0, 0.0

        hits_1 = 0
        hits_5 = 0
        for i in range(n):
            query_label = labels[i]
            # Rank predictions by descending similarity
            ranked_indices = np.argsort(-sim_matrix[i])
            top_1_label = labels[ranked_indices[0]]
            top_5_labels = [labels[idx] for idx in ranked_indices[:min(5, n)]]

            if top_1_label == query_label:
                hits_1 += 1
            if query_label in top_5_labels:
                hits_5 += 1

        return hits_1 / n, hits_5 / n
