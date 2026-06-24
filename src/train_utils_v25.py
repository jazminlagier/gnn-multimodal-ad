#!/usr/bin/env python3
"""
train_utils_v25.py
==================
- Unified training utilities for GCN/GAT/GKAN
- Evaluation: ROC-AUC, PR-AUC, Acc, F1, Precision/Recall, Specificity, Brier, ECE
- Class-imbalance handling
- Cosine restarts for GKAN, ReduceLROnPlateau for GCN/GAT
- Temperature scaling (fit_temperature) for post-hoc calibration on the val set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, accuracy_score,
        brier_score_loss, confusion_matrix, balanced_accuracy_score,
        average_precision_score,
        precision_recall_fscore_support,
    )
import os
import matplotlib.pyplot as plt

# Imports for training-curve tracking and interpretability
from pathlib import Path
from typing import Dict, Optional
from torch_geometric.loader import DataLoader
import logging

logger = logging.getLogger(__name__)


# -----------------------------
# Losses and helpers
# -----------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # tensor [C] or None
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n = pred.size(-1)
        true = torch.zeros_like(pred)
        true.scatter_(1, target.unsqueeze(1), 1)
        true = true * (1 - self.smoothing) + self.smoothing / n
        return -(true * F.log_softmax(pred, dim=-1)).sum(dim=1).mean()

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    """
    y_true: 1D array-like of {0,1}
    y_prob: 1D array-like of probabilities for the positive class (float in [0,1])
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    eps = 1e-12
    sensitivity = tp / (tp + fn + eps)  # recall for positive class
    specificity = tn / (tn + fp + eps)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    bal_acc = 0.5 * (sensitivity + specificity)

    auc = roc_auc_score(y_true, y_prob)
    try:
        ap = average_precision_score(y_true, y_prob)
    except Exception:
        ap = float("nan")

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "ap": ap,
       #"threshold": float(threshold),
    }


def _ece(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    probs_pos = probs[:, 1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (probs_pos >= bins[i]) & (probs_pos < bins[i + 1])
        if m.sum() == 0:
            continue
        conf = probs_pos[m].mean()
        acc = (labels[m] == (probs_pos[m] >= 0.5)).mean()
        ece += np.abs(acc - conf) * (m.sum() / len(probs_pos))
    return float(ece)

def train_model_with_tracking(
    model,
    model_name: str,
    train_loader,
    val_loader,
    config: dict,
    device,
    learning_curve_tracker=None,
    gradient_analyzer=None,
    save_dir=None
):
    """
    Enhanced training function with comprehensive tracking.

    Args:
        model: GNN model
        model_name: Name of model architecture
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
        device: torch device
        learning_curve_tracker: LearningCurveTracker instance
        gradient_analyzer: GradientFlowAnalyzer instance
        save_dir: Directory to save checkpoints (Path or str)

    Returns:
        Dictionary with training info
    """
    from learning_curve_tracker import LearningCurveTracker, GradientFlowAnalyzer
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    # Initialize trackers if not provided
    if learning_curve_tracker is None:
        learning_curve_tracker = LearningCurveTracker()
    if gradient_analyzer is None:
        gradient_analyzer = GradientFlowAnalyzer()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training setup
    epochs = config['epochs']
    lr = config['learning_rate']
    wd = config['weight_decay']
    patience = config.get('early_stopping_patience', 50)
    focal_gamma = config.get('focal_gamma', 2.0)
    label_smoothing = config.get('label_smoothing', 0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    best_val_loss = float('inf')
    best_val_auc = 0.0
    epochs_no_improve = 0
    best_model_state = None

    logger.info(f"Training {model_name} with learning curve tracking...")

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)

            # Focal loss with label smoothing
            y_smooth = batch.y.float()
            if label_smoothing > 0:
                y_smooth = y_smooth * (1 - label_smoothing) + 0.5 * label_smoothing

            ce_loss = F.cross_entropy(out, batch.y, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** focal_gamma * ce_loss).mean()

            focal_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += focal_loss.item() * batch.num_graphs

            with torch.no_grad():
                pred = out.argmax(dim=1)
                train_preds.extend(pred.cpu().numpy())
                train_labels.extend(batch.y.cpu().numpy())

        train_loss /= len(train_loader.dataset)

        # Train metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_bal_acc = balanced_accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)

        # Get probabilities for AUC
        model.eval()
        train_probs = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                out = model(batch)
                probs = F.softmax(out, dim=1)[:, 1]
                train_probs.extend(probs.cpu().numpy())

        train_auc = roc_auc_score(train_labels, train_probs) if len(np.unique(train_labels)) > 1 else 0.5

        # Validation
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_labels = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)

                loss = F.cross_entropy(out, batch.y)
                val_loss += loss.item() * batch.num_graphs

                probs = F.softmax(out, dim=1)[:, 1]
                pred = out.argmax(dim=1)

                val_probs.extend(probs.cpu().numpy())
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Val metrics
        val_acc = accuracy_score(val_labels, val_preds)
        val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5

        # Calculate sensitivity and specificity
        val_preds_arr = np.array(val_preds)
        val_labels_arr = np.array(val_labels)

        tp = ((val_preds_arr == 1) & (val_labels_arr == 1)).sum()
        tn = ((val_preds_arr == 0) & (val_labels_arr == 0)).sum()
        fp = ((val_preds_arr == 1) & (val_labels_arr == 0)).sum()
        fn = ((val_preds_arr == 0) & (val_labels_arr == 1)).sum()

        val_sensitivity = tp / (tp + fn + 1e-12)
        val_specificity = tn / (tn + fp + 1e-12)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record gradients
        gradient_analyzer.record_gradients(model, epoch)

        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        grad_norm = total_norm ** 0.5

        # Update learning curve tracker
        learning_curve_tracker.update(epoch, {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_balanced_accuracy': train_bal_acc,
            'train_f1': train_f1,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_balanced_accuracy': val_bal_acc,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'val_sensitivity': val_sensitivity,
            'val_specificity': val_specificity,
            'learning_rate': current_lr,
            'grad_norm': grad_norm
        })

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()

            # Save best model checkpoint
            if save_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'val_loss': best_val_loss
                }, save_dir / f'best_model_epoch_{epoch}.pth')

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: Loss={train_loss:.4f} AUC={train_auc:.4f} | "
                f"Val: Loss={val_loss:.4f} AUC={val_auc:.4f} BalAcc={val_bal_acc:.4f} | "
                f"LR={current_lr:.6f}"
            )

        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Save final model
    if save_dir:
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_epoch': epoch,
            'best_val_auc': best_val_auc,
            'config': config
        }, save_dir / 'final_model.pth')

        # Save learning curves
        learning_curve_tracker.plot_learning_curves(
            save_dir / 'learning_curves.png',
            title=f'{model_name} Learning Curves'
        )
        learning_curve_tracker.save_history(save_dir / 'training_history.json')

        # Save gradient flow
        gradient_analyzer.plot_gradient_flow(save_dir / 'gradient_flow.png')

    return {
        'best_val_auc': float(best_val_auc),
        'best_val_loss': float(best_val_loss),
        'final_epoch': int(epoch),
        'learning_curve_tracker': learning_curve_tracker,
        'gradient_analyzer': gradient_analyzer
    }


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_model_with_interpretability(model, model_name, test_loader, device,
                                         threshold=0.5, temperature=1.0,
                                         run_interpretability=True, output_dir=None):
    """Evaluate model and run interpretability analysis"""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  accuracy_score, f1_score, confusion_matrix)

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    # Collect node features
    all_node_features = []  # List of [num_nodes, num_features] per sample
    all_edge_indices = []
    all_batch_assignments = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits / temperature, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(batch.y.cpu())

            # Store features per graph
            if run_interpretability:
                # Get per-graph node features
                batch_ptr = batch.ptr if hasattr(batch, 'ptr') else None

                if batch_ptr is not None:
                    # Process each graph in the batch separately
                    for i in range(len(batch_ptr) - 1):
                        start_idx = batch_ptr[i]
                        end_idx = batch_ptr[i + 1]

                        graph_x = batch.x[start_idx:end_idx].cpu().numpy()  # [num_nodes, num_features]
                        all_node_features.append(graph_x)
                else:
                    # Single graph - fallback
                    all_node_features.append(batch.x.cpu().numpy())

    # Concatenate predictions
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    pr_auc = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    metrics = {
        'auc': float(auc),
        'pr_auc': float(pr_auc),
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_accuracy),
        'f1': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'threshold': float(threshold),
        'temperature': float(temperature)
    }

    # Run interpretability if requested
    if run_interpretability and output_dir:
        interp_dir = Path(output_dir) / 'interpretability'
        interp_dir.mkdir(parents=True, exist_ok=True)

        try:
            from dk_interpretability import EnhancedGNNInterpretability

            analyzer = EnhancedGNNInterpretability(model, device)

            # Integrated-gradients node importance. Stored under the
            # 'integrated_gradients' key, which is the structure the
            # cross-fold aggregation step in the runner expects.
            ig_results = analyzer.compute_node_importance_integrated_gradients(test_loader)
            importance_results = {'integrated_gradients': ig_results}

            import pickle
            with open(interp_dir / 'importance_results.pkl', 'wb') as f:
                pickle.dump(importance_results, f)

            # Visualization is best-effort and must not break evaluation.
            if 'node_importance_mean' in ig_results:
                try:
                    analyzer.create_comprehensive_visualization(
                        interp_dir / 'interpretability_summary.png'
                    )
                except Exception as viz_err:
                    print(f"      Interpretability visualization skipped: {viz_err}")

            metrics['interpretability_dir'] = str(interp_dir)
            print(f"      Interpretability saved to {interp_dir}")

        except Exception as e:
            print(f"      Interpretability failed: {e}")
            import traceback
            traceback.print_exc()
            metrics['interpretability_dir'] = None

    return metrics
# === PATCH: FULL REPLACEMENT evaluate_model ===
def evaluate_model(model, model_type, data_loader, device, threshold=0.5, temperature=None):
    """
    Multiclass-aware evaluation with validation loss.
    - If C==2: threshold on class-1 prob (binary metrics incl. SEN/SPE).
    - If C>=3: argmax; report macro metrics incl. macro SEN/SPE.
    Returns a dict including 'val_loss'.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, accuracy_score,
        brier_score_loss, confusion_matrix, balanced_accuracy_score,
        precision_recall_fscore_support,
    )
    from sklearn.preprocessing import label_binarize

    model.eval()
    probs_full_all, labels_all = [], []

    # --- accumulate validation loss over batches ---
    total_val_loss, n_batches = 0.0, 0
    ce = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(batch)
            if temperature is not None:
                logits = logits / max(1e-3, float(temperature))

            # val CE loss
            total_val_loss += float(ce(logits, batch.y).item())
            n_batches += 1

            probs = F.softmax(logits, dim=1).detach().cpu().numpy()  # [B,C]
            y = batch.y.detach().cpu().numpy().astype(int)
            probs_full_all.append(probs)
            labels_all.append(y)

    if n_batches == 0:
        # Degenerate edge case
        return {
            "auc": float("nan"),
            "pr_auc": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "val_loss": float("nan"),
        }

    probs_full = np.vstack(probs_full_all)           # [N,C]
    y_true = np.concatenate(labels_all)              # [N]
    C = probs_full.shape[1]
    val_loss = total_val_loss / max(1, n_batches)

    # ----- Binary case -----
    if C == 2:
        y_prob = probs_full[:, 1]
        y_pred = (y_prob >= float(threshold)).astype(np.int64)

        # Metrics
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.5
        pr_auc = average_precision_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred) if (y_pred.sum() > 0) else 0.0

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)

        brier = brier_score_loss(y_true, y_prob)
        # ECE (binary): bins on positive class prob
        n_bins = 15
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if m.sum() == 0:
                continue
            acc_bin = (y_pred[m] == y_true[m]).mean()
            conf_bin = y_prob[m].mean()
            ece += (m.mean()) * abs(acc_bin - conf_bin)

        return {
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            "accuracy": float(acc),
            "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
            "f1": float(f1),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "brier": float(brier),
            "ece": float(ece),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "val_loss": float(val_loss),
        }

    # ----- Multiclass case (C >= 3) -----
    # Predictions by argmax
    y_pred = probs_full.argmax(axis=1)

    # Macro metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # AUC/PR-AUC macro (one-vs-rest)
    classes = np.unique(y_true)
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=classes)  # [N, C']
    # Align probs to classes order
    probs_for_auc = probs_full[:, classes]

    # Guard against degenerate folds (e.g., missing a class)
    try:
        auc_macro = roc_auc_score(y_true_bin, probs_for_auc, average='macro', multi_class='ovr')
    except ValueError:
        auc_macro = 0.5

    try:
        pr_auc_macro = average_precision_score(y_true_bin, probs_for_auc, average='macro')
    except ValueError:
        pr_auc_macro = 0.0

    # Per-class sensitivity (recall) and specificity, then macro-averaged
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + fn)

    sens_per_class = tp / (tp + fn + 1e-12)
    spec_per_class = tn / (tn + fp + 1e-12)
    sensitivity_macro = sens_per_class.mean()
    specificity_macro = spec_per_class.mean()

    # Multiclass Brier: mean over classes of squared error
    y_onehot = np.zeros_like(probs_for_auc)
    for i, c in enumerate(classes):
        y_onehot[:, i] = (y_true == c).astype(float)
    brier_multi = float(np.mean(np.sum((probs_for_auc - y_onehot)**2, axis=1)))

    # Multiclass ECE (confidence/accuracy bins on max prob)
    n_bins = 15
    conf = probs_full.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1])
        if m.sum() == 0:
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        ece += (m.mean()) * abs(acc_bin - conf_bin)

    return {
        "auc": float(auc_macro),
        "pr_auc": float(pr_auc_macro),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1": float(f1_macro),
        "sensitivity": float(sensitivity_macro),   # macro recall
        "specificity": float(specificity_macro),   # macro specificity
        "brier": float(brier_multi),
        "ece": float(ece),
        "val_loss": float(val_loss),
        # TP/FP/TN/FN are not single numbers in multiclass; keep keys absent.
    }



# -----------------------------
# Training loops
# -----------------------------

def _compute_class_weights_from_loader(train_loader, device):
    labels = []
    for b in train_loader:
        labels.extend(b.y.cpu().numpy())
    counts = np.bincount(labels)
    # inverse frequency, normalized to sum to 2.0 (so ~[w0,w1] scaled but stable)
    weights = (counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.sum() * 2.0
    return torch.FloatTensor(weights).to(device)


def train_model(model, model_type, train_loader, val_loader, config, device):
    """
    For GCN/GAT:
      - Weighted CrossEntropy
      - ReduceLROnPlateau on val AUC
      - Early stopping, restore best
    """
    class_weights = _compute_class_weights_from_loader(train_loader, device)
    gamma = float(config.get('focal_gamma', 2.0))
    criterion = FocalLoss(alpha=class_weights, gamma=gamma)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(config['epochs']):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_metrics = evaluate_model(model, model_type, val_loader, device, threshold=0.5)
        scheduler.step(val_metrics['auc'])

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return {'best_val_auc': float(best_auc)}

def train_model_optimized(model, model_type, train_loader, val_loader, config, device):
    """
    For GKAN:
      - Focal loss with class weighting
      - Label smoothing
      - CosineAnnealingWarmRestarts
      - Optional spline reg if model exposes compute_spline_regularization()
    """
    class_weights = _compute_class_weights_from_loader(train_loader, device)
    gamma = float(config.get('focal_gamma', 2.0))
    ls = float(config.get('label_smoothing', 0.1))
    criterion_focal = FocalLoss(alpha=class_weights, gamma=gamma)
    criterion_ls = LabelSmoothingCrossEntropy(smoothing=ls)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    train_losses_per_epoch = []
    val_losses_per_epoch = []

    best_auc, best_state, best_metrics, patience = 0.0, None, None, 0

    # optional: where to save learning curve
    out_dir = config.get("out_dir", os.path.join(os.getcwd(), "runs"))
    os.makedirs(out_dir, exist_ok=True)
    model_name = config.get("model_name", model_type)

    for epoch in range(config['epochs']):
        model.train()
        # --- accumulate train loss this epoch ---
        train_loss_sum, train_batches = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)

            loss_focal = criterion_focal(logits, batch.y)
            loss_ls = criterion_ls(logits, batch.y)
            loss = 0.7 * loss_focal + 0.3 * loss_ls

            if hasattr(model, "compute_spline_regularization"):
                loss = loss + 1e-5 * model.compute_spline_regularization()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # --- record train loss ---
            train_loss_sum += float(loss.item())
            train_batches += 1

        scheduler.step(epoch)  # epoch step for warm restarts

        # --- per-epoch averaged train loss ---
        avg_train_loss = train_loss_sum / max(1, train_batches)
        train_losses_per_epoch.append(avg_train_loss)

        # --- validation (now returns val_loss + sen/spe, etc.) ---
        val_metrics = evaluate_model(model, model_type, val_loader, device, threshold=0.5)
        val_losses_per_epoch.append(val_metrics["val_loss"])

        # --- early stopping keyed on AUC (as before) ---
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = dict(val_metrics)  # keep the whole metrics snapshot
            patience = 0
        else:
            patience += 1
            if patience >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}. GKAN best val AUC={best_auc:.4f}")
                break

    # restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # --- save learning curve plot ---
    lc_path = os.path.join(out_dir, f"learning_curve_{model_name}.png")
    try:
        plt.figure()
        plt.plot(range(1, len(train_losses_per_epoch) + 1), train_losses_per_epoch, label="train")
        plt.plot(range(1, len(val_losses_per_epoch) + 1), val_losses_per_epoch, label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"Learning Curve - {model_name}")
        plt.legend(); plt.tight_layout()
        plt.savefig(lc_path, dpi=200)
        plt.close()
    except Exception as e:
        lc_path = None
        print(f"[warn] could not save learning curve: {e}")

    # --- return includes best AUC + best SEN/SPE + learning curve info ---
    ret = {
        'best_val_auc': float(best_auc),
        'train_losses': train_losses_per_epoch,
        'val_losses': val_losses_per_epoch,
        'learning_curve_png': lc_path,
    }
    if isinstance(best_metrics, dict):
        ret.update({
            'best_val_ap': best_metrics.get('ap'),
            'best_val_acc': best_metrics.get('acc'),
            'best_val_bal_acc': best_metrics.get('bal_acc'),
            'best_val_f1': best_metrics.get('f1'),
            'best_val_prec': best_metrics.get('precision'),
            'best_val_rec': best_metrics.get('recall'),
            'best_val_sen': best_metrics.get('sensitivity'),  # <-- SEN
            'best_val_spe': best_metrics.get('specificity'),  # <-- SPE
            'best_val_loss': best_metrics.get('val_loss'),
        })
    return ret


# -----------------------------
# Temperature scaling
# -----------------------------

def fit_temperature(model, val_loader, device, max_iter=50, init_T=1.0):
    """
    Fit a single temperature scalar T on validation logits to minimize NLL.
    Returns T (float). Does not modify the model.
    """
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch).detach()
            logits_list.append(logits)
            labels_list.append(batch.y.detach())

    if len(logits_list) == 0:
        return 1.0

    logits = torch.cat(logits_list, dim=0).to(device)
    labels = torch.cat(labels_list, dim=0).to(device)

    T = nn.Parameter(torch.tensor(float(init_T), device=device))
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    nll_criterion = nn.CrossEntropyLoss()

    def _eval():
        optimizer.zero_grad()
        scaled = logits / T.clamp_min(1e-3)
        loss = nll_criterion(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(_eval)
    return float(T.detach().cpu().item())


def apply_temperature(logits, temperature: float):
    """
    Apply a scalar temperature to logits.
    """
    if temperature is None:
        return logits
    t = max(1e-3, float(temperature))
    return logits / t

def tune_threshold_on_val(model, val_loader, device, temperature: float = None,
                          metric: str = "balanced_acc", grid=None,
                          min_sensitivity: float = None,
                          min_specificity: float = None,
                          prevalence_tolerance: float = None):
    """
    Binary-only: grid-search the decision threshold on the validation set with optional constraints.
      - metric: "balanced_acc" (default) or "youden" or "f1"
      - min_sensitivity / min_specificity: require TPR/TNR >= given values
      - prevalence_tolerance: require the predicted positive rate to be within
        +/- tolerance of the validation prevalence (absolute, e.g., 0.15)
    Returns (best_threshold, info_dict)
    """

    model.eval()
    if grid is None:
        grid = np.linspace(0.02, 0.98, 193)

    # ---------- SAFETY GUARD: skip if labels are multiclass ----------
    labels_probe = []
    with torch.no_grad():
        for batch in val_loader:
            labels_probe.append(batch.y.detach().cpu().numpy())
    labels_probe = np.concatenate(labels_probe)
    if len(np.unique(labels_probe)) > 2:
        # Multiclass threshold tuning not applicable
        return 0.50, {"skipped": True, "reason": "multiclass"}
    # ----------------------------------------------------------------

    probs_all, labels_all = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch)
            logits = apply_temperature(logits, temperature)
            p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # prob of positive class
            y = batch.y.detach().cpu().numpy()
            probs_all.append(p)
            labels_all.append(y)

    probs_all = np.concatenate(probs_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0).astype(int)
    val_prev = float(labels_all.mean())

    best_t = 0.50
    best_score = -np.inf
    best_stats = None

    for t in grid:
        preds = (probs_all >= t).astype(np.int64)

        # Skip degenerate thresholds that predict only one class
        if preds.max() == preds.min():
            continue

        tp = int(((preds == 1) & (labels_all == 1)).sum())
        tn = int(((preds == 0) & (labels_all == 0)).sum())
        fp = int(((preds == 1) & (labels_all == 0)).sum())
        fn = int(((preds == 0) & (labels_all == 1)).sum())

        sens = tp / (tp + fn + 1e-12)   # TPR / recall
        spec = tn / (tn + fp + 1e-12)   # TNR
        acc  = (tp + tn) / (tp + tn + fp + fn + 1e-12)
        bal_acc = 0.5 * (sens + spec)

        # prevalence constraint (optional)
        if prevalence_tolerance is not None:
            pred_prev = preds.mean()
            if abs(pred_prev - val_prev) > float(prevalence_tolerance):
                continue

        # sensitivity/specificity constraints (optional)
        if (min_sensitivity is not None) and (sens < float(min_sensitivity)):
            continue
        if (min_specificity is not None) and (spec < float(min_specificity)):
            continue

        # choose score
        if metric == "youden":
            score = sens + spec - 1.0  # Youden's J
        elif metric == "f1":
            precision = tp / (tp + fp + 1e-12)
            f1 = 2 * precision * sens / (precision + sens + 1e-12)
            score = f1
        else:  # "balanced_acc" (default)
            score = bal_acc

        if score > best_score:
            best_score = score
            best_t = float(t)
            best_stats = {
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "sensitivity": float(sens),
                "specificity": float(spec),
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "val_prevalence": float(val_prev),
                "pred_prevalence": float(preds.mean()),
                "metric": metric,
                "score": float(score),
            }

    # Clamp to safe range (only relevant in binary usage)
    if not np.isfinite(best_t) or best_t <= 0.0:
        best_t = 0.05
    if best_t >= 0.95:
        best_t = 0.90

    return best_t, (best_stats or {"warn": "no valid threshold found"})


def wilson_ci(successes, n, alpha=0.05):
    from scipy.stats import norm
    if n == 0: return (np.nan, np.nan)
    z = norm.ppf(1 - alpha/2); p = successes / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    half = z * np.sqrt((p*(1-p) + z*z/(4*n))/n)
    lo, hi = (centre - half)/denom, (centre + half)/denom
    return float(max(0.0, lo)), float(min(1.0, hi))

def ci_sens_spec(tp, fp, tn, fn, alpha=0.05):
    sens_lo, sens_hi = wilson_ci(tp, tp+fn, alpha)
    spec_lo, spec_hi = wilson_ci(tn, tn+fp, alpha)
    return (sens_lo, sens_hi), (spec_lo, spec_hi)

def bootstrap_ci_metric(labels, scores_or_preds, metric='BA', n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(labels); x = np.asarray(scores_or_preds)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), size=len(y))
        yb, xb = y[idx], x[idx]
        if metric == 'BA':
            pb = (xb >= 0.5).astype(int)
            vals.append(balanced_accuracy_score(yb, pb))
        elif metric == 'F1':
            pb = (xb >= 0.5).astype(int)
            vals.append(f1_score(yb, pb, zero_division=0))
        elif metric == 'AUC':
            if yb.min() == yb.max(): continue
            vals.append(roc_auc_score(yb, xb))
    if not vals: return (np.nan, np.nan)
    lo = np.percentile(vals, 100*alpha/2); hi = np.percentile(vals, 100*(1-alpha/2))
    return float(lo), float(hi)

    return best_t, best_metrics
