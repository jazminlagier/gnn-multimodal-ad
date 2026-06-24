"""
Learning Curve Tracker for GNN Training
========================================
Tracks and visualizes:
- Training/validation loss curves
- Metric evolution (AUC, accuracy, balanced accuracy, F1)
- Gradient flow analysis
- Learning rate schedules
- Overfitting detection
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class LearningCurveTracker:
    """
    Track and visualize learning curves during training.
    """

    def __init__(self):
        self.history = defaultdict(list)
        self.epoch_data = []

    def update(self, epoch: int, metrics: Dict):
        """
        Update tracker with metrics from current epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics (loss, auc, accuracy, etc.)
        """
        self.epoch_data.append({
            'epoch': epoch,
            **metrics
        })

        for key, value in metrics.items():
            self.history[key].append(value)

    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric"""
        return self.history.get(metric_name, [])

    def detect_overfitting(self, patience: int = 10) -> Dict:
        """
        Detect overfitting by comparing train and validation metrics.

        Returns:
            Dictionary with overfitting indicators
        """
        if 'train_loss' not in self.history or 'val_loss' not in self.history:
            return {'overfitting_detected': False}

        train_loss = np.array(self.history['train_loss'])
        val_loss = np.array(self.history['val_loss'])

        # Check if validation loss increasing while training loss decreasing
        recent_epochs = min(patience, len(train_loss))

        if recent_epochs < 5:
            return {'overfitting_detected': False}

        train_trend = train_loss[-recent_epochs:].mean() - train_loss[-recent_epochs:-recent_epochs//2].mean()
        val_trend = val_loss[-recent_epochs:].mean() - val_loss[-recent_epochs:-recent_epochs//2].mean()

        overfitting = bool(train_trend < 0 and val_trend > 0)  # cast to Python bool
        gap = val_loss[-1] - train_loss[-1]

        return {
            'overfitting_detected': bool(overfitting),  # cast to Python bool
            'train_val_gap': float(gap),
            'train_trend': float(train_trend),
            'val_trend': float(val_trend),
            'recommendation': 'Consider early stopping or regularization' if overfitting else 'Training is stable'
        }

    def find_best_epoch(self, metric: str = 'val_auc', mode: str = 'max') -> Dict:
        """
        Find the epoch with best performance.

        Args:
            metric: Metric to optimize
            mode: 'max' or 'min'

        Returns:
            Dictionary with best epoch info
        """
        if metric not in self.history:
            return {}

        values = np.array(self.history[metric])

        if mode == 'max':
            best_idx = values.argmax()
        else:
            best_idx = values.argmin()

        return {
            'best_epoch': int(best_idx + 1),
            'best_value': float(values[best_idx]),
            'improvement_from_start': float(values[best_idx] - values[0]) if mode == 'max' else float(values[0] - values[best_idx]),
            'epochs_since_best': int(len(values) - best_idx - 1)
        }

    def plot_learning_curves(self, output_path: Path, title: str = "Learning Curves"):
        """
        Create comprehensive learning curve visualization.
        """
        logger.info(f"Plotting learning curves: {title}")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        epochs = np.arange(1, len(self.epoch_data) + 1)

        # 1. loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        if 'train_loss' in self.history:
            ax1.plot(epochs, self.history['train_loss'],
                    label='Train Loss', linewidth=2, color='#1f77b4', marker='o', markersize=3)
        if 'val_loss' in self.history:
            ax1.plot(epochs, self.history['val_loss'],
                    label='Val Loss', linewidth=2, color='#ff7f0e', marker='s', markersize=3)

        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Loss Curves', fontweight='bold', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. auc curves
        ax2 = fig.add_subplot(gs[0, 1])
        if 'train_auc' in self.history:
            ax2.plot(epochs, self.history['train_auc'],
                    label='Train AUC', linewidth=2, color='#2ca02c', marker='o', markersize=3)
        if 'val_auc' in self.history:
            ax2.plot(epochs, self.history['val_auc'],
                    label='Val AUC', linewidth=2, color='#d62728', marker='s', markersize=3)

        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('AUC', fontweight='bold')
        ax2.set_title('AUC Curves', fontweight='bold', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.5, 1.0])

        # 3. balanced accuracy
        ax3 = fig.add_subplot(gs[0, 2])
        if 'train_balanced_accuracy' in self.history:
            ax3.plot(epochs, self.history['train_balanced_accuracy'],
                    label='Train Bal Acc', linewidth=2, color='#9467bd', marker='o', markersize=3)
        if 'val_balanced_accuracy' in self.history:
            ax3.plot(epochs, self.history['val_balanced_accuracy'],
                    label='Val Bal Acc', linewidth=2, color='#8c564b', marker='s', markersize=3)

        ax3.set_xlabel('Epoch', fontweight='bold')
        ax3.set_ylabel('Balanced Accuracy', fontweight='bold')
        ax3.set_title('Balanced Accuracy Curves', fontweight='bold', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.5, 1.0])

        # 4. f1 score
        ax4 = fig.add_subplot(gs[1, 0])
        if 'train_f1' in self.history:
            ax4.plot(epochs, self.history['train_f1'],
                    label='Train F1', linewidth=2, color='#e377c2', marker='o', markersize=3)
        if 'val_f1' in self.history:
            ax4.plot(epochs, self.history['val_f1'],
                    label='Val F1', linewidth=2, color='#7f7f7f', marker='s', markersize=3)

        ax4.set_xlabel('Epoch', fontweight='bold')
        ax4.set_ylabel('F1 Score', fontweight='bold')
        ax4.set_title('F1 Score Curves', fontweight='bold', fontsize=12)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # 5. sensitivity & specificity
        ax5 = fig.add_subplot(gs[1, 1])
        if 'val_sensitivity' in self.history and 'val_specificity' in self.history:
            ax5.plot(epochs, self.history['val_sensitivity'],
                    label='Sensitivity', linewidth=2, color='#bcbd22', marker='o', markersize=3)
            ax5.plot(epochs, self.history['val_specificity'],
                    label='Specificity', linewidth=2, color='#17becf', marker='s', markersize=3)

            ax5.set_xlabel('Epoch', fontweight='bold')
            ax5.set_ylabel('Score', fontweight='bold')
            ax5.set_title('Sensitivity & Specificity', fontweight='bold', fontsize=12)
            ax5.legend(loc='best')
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([0.0, 1.0])

        # 6. learning rate
        ax6 = fig.add_subplot(gs[1, 2])
        if 'learning_rate' in self.history:
            ax6.plot(epochs, self.history['learning_rate'],
                    linewidth=2, color='#ff9896', marker='o', markersize=3)
            ax6.set_xlabel('Epoch', fontweight='bold')
            ax6.set_ylabel('Learning Rate', fontweight='bold')
            ax6.set_title('Learning Rate Schedule', fontweight='bold', fontsize=12)
            ax6.set_yscale('log')
            ax6.grid(True, alpha=0.3)

        # 7. train-val gap
        ax7 = fig.add_subplot(gs[2, 0])
        if 'train_loss' in self.history and 'val_loss' in self.history:
            gap = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
            ax7.plot(epochs, gap, linewidth=2, color='#d62728', marker='o', markersize=3)
            ax7.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax7.fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='red', label='Overfitting')
            ax7.fill_between(epochs, 0, gap, where=(gap <= 0), alpha=0.3, color='green', label='Underfitting')

            ax7.set_xlabel('Epoch', fontweight='bold')
            ax7.set_ylabel('Val Loss - Train Loss', fontweight='bold')
            ax7.set_title('Generalization Gap', fontweight='bold', fontsize=12)
            ax7.legend(loc='best')
            ax7.grid(True, alpha=0.3)

        # 8. gradient norm (if available)
        ax8 = fig.add_subplot(gs[2, 1])
        if 'grad_norm' in self.history:
            ax8.plot(epochs, self.history['grad_norm'],
                    linewidth=2, color='#9467bd', marker='o', markersize=3)
            ax8.set_xlabel('Epoch', fontweight='bold')
            ax8.set_ylabel('Gradient Norm', fontweight='bold')
            ax8.set_title('Gradient Magnitude', fontweight='bold', fontsize=12)
            ax8.grid(True, alpha=0.3)

        # 9. best epoch indicator
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        # Find best epoch
        best_info = self.find_best_epoch('val_auc', 'max')
        overfit_info = self.detect_overfitting()

        info_text = "Training Summary:\n\n"

        if best_info:
            info_text += f"Best Val AUC: {best_info['best_value']:.4f}\n"
            info_text += f"Best Epoch: {best_info['best_epoch']}\n"
            info_text += f"Epochs Since Best: {best_info['epochs_since_best']}\n\n"

        info_text += f"Overfitting: {'WARNING Yes' if overfit_info['overfitting_detected'] else 'OK No'}\n"

        if overfit_info['overfitting_detected']:
            info_text += f"Train-Val Gap: {overfit_info['train_val_gap']:.4f}\n"
            info_text += f"\n{overfit_info['recommendation']}"

        ax9.text(0.5, 0.5, info_text,
                transform=ax9.transAxes,
                fontsize=11,
                verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Learning curves saved to: {output_path}")

    def save_history(self, output_path: Path):
        """Save training history to JSON"""

        def convert_to_serializable(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj

        history_dict = {
            'epochs': list(range(1, len(self.epoch_data) + 1)),
            **{k: [float(x) for x in v] for k, v in self.history.items()},
            'best_epoch_info': convert_to_serializable(self.find_best_epoch('val_auc', 'max')),
            'overfitting_analysis': convert_to_serializable(self.detect_overfitting())
        }

        with open(output_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"Training history saved to: {output_path}")
    def plot_metric_comparison(self, metrics: List[str], output_path: Path,
                              title: str = "Metric Comparison"):
        """
        Plot multiple metrics on same axes for comparison.

        Args:
            metrics: List of metric names to plot
            output_path: Where to save plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        epochs = np.arange(1, len(self.epoch_data) + 1)
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

        for metric, color in zip(metrics, colors):
            if metric in self.history:
                ax.plot(epochs, self.history[metric],
                       label=metric.replace('_', ' ').title(),
                       linewidth=2, color=color, marker='o', markersize=3)

        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Value', fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Metric comparison saved to: {output_path}")


class GradientFlowAnalyzer:
    """
    Analyze gradient flow through network layers.
    Helps detect vanishing/exploding gradients.
    """

    def __init__(self):
        self.gradient_norms = defaultdict(list)

    def record_gradients(self, model, epoch: int):
        """
        Record gradient norms for each layer.

        Args:
            model: PyTorch model
            epoch: Current epoch
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.gradient_norms[name].append({
                    'epoch': epoch,
                    'norm': grad_norm
                })

    def plot_gradient_flow(self, output_path: Path):
        """
        Visualize gradient flow across layers and epochs.
        """
        if not self.gradient_norms:
            logger.warning("No gradient data recorded")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Gradient norms over time for each layer
        for layer_name, grad_data in self.gradient_norms.items():
            epochs = [d['epoch'] for d in grad_data]
            norms = [d['norm'] for d in grad_data]
            ax1.plot(epochs, norms, label=layer_name, linewidth=1.5, alpha=0.7)

        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Gradient Norm', fontweight='bold')
        ax1.set_title('Gradient Flow Over Training', fontweight='bold', fontsize=12)
        ax1.set_yscale('log')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Final gradient distribution
        final_norms = []
        layer_names = []

        for layer_name, grad_data in self.gradient_norms.items():
            if grad_data:
                final_norms.append(grad_data[-1]['norm'])
                layer_names.append(layer_name.split('.')[-1][:15])  # Shorten names

        ax2.barh(range(len(layer_names)), final_norms, alpha=0.7, edgecolor='black')
        ax2.set_yticks(range(len(layer_names)))
        ax2.set_yticklabels(layer_names, fontsize=8)
        ax2.set_xlabel('Final Gradient Norm', fontweight='bold')
        ax2.set_title('Final Gradient Distribution by Layer', fontweight='bold', fontsize=12)
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Gradient flow visualization saved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Learning curve tracker module loaded")