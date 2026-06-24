"""
Enhanced Graph Neural Network Interpretability for AD Classification
====================================================================
Comprehensive analysis including:
- Node-level importance (which brain regions matter most)
- Feature-level importance (which modalities contribute)
- Attention weight analysis (for GAT models)
- Layer-wise gradient flow
- Model weight saving and extraction
- Learning curve tracking
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)


class EnhancedGNNInterpretability:
    """
    Comprehensive interpretability suite for GNN-based AD classification.
    """

    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Trained GNN model
            device: torch device
        """
        self.model = model
        self.device = device
        self.model.eval()

        # DK region names (112 regions)
        self.region_names = self._get_dk_region_names()

        # AD-relevant regions for comparison
        self.ad_pathology_regions = self._define_ad_pathology_regions()

        # Storage for interpretability results
        self.importance_cache = {}

    def _forward(self, data):
        """
        Call the model in a signature-agnostic way.
        Prefers model(data); falls back to model(x, edge_index[, batch]).
        """
        try:
            return self.model(data)
        except TypeError:
            x = data.x
            edge_index = data.edge_index
            batch_idx = getattr(data, "batch", None)
            if batch_idx is None:
                return self.model(x, edge_index)
            return self.model(x, edge_index, batch_idx)


    def _get_dk_region_names(self) -> List[str]:
        """Get Desikan-Killiany region names"""

        # Left hemisphere cortical
        lh_cortical = [
            "L_BanksSTS", "L_CaudalAnteriorCingulate", "L_CaudalMiddleFrontal",
            "L_Cuneus", "L_Entorhinal", "L_FrontalPole", "L_Fusiform",
            "L_InferiorParietal", "L_InferiorTemporal", "L_Insula",
            "L_IsthmusCingulate", "L_LateralOccipital", "L_LateralOrbitofrontal",
            "L_Lingual", "L_MedialOrbitofrontal", "L_MiddleTemporal",
            "L_Paracentral", "L_Parahippocampal", "L_ParsOpercularis",
            "L_ParsOrbitalis", "L_ParsTriangularis", "L_Pericalcarine",
            "L_Postcentral", "L_PosteriorCingulate", "L_Precentral",
            "L_Precuneus", "L_RostralAnteriorCingulate", "L_RostralMiddleFrontal",
            "L_SuperiorFrontal", "L_SuperiorParietal", "L_SuperiorTemporal",
            "L_Supramarginal", "L_TemporalPole", "L_TransverseTemporal"
        ]

        # Right hemisphere cortical
        rh_cortical = [r.replace("L_", "R_") for r in lh_cortical]

        # Subcortical structures
        subcortical = [
            "L_Hippocampus", "R_Hippocampus",
            "L_Amygdala", "R_Amygdala",
            "L_Caudate", "R_Caudate",
            "L_Putamen", "R_Putamen",
            "L_Pallidum", "R_Pallidum",
            "L_Thalamus", "R_Thalamus",
            "L_Accumbens", "R_Accumbens"
        ]

        # Combine (should be 112 total)
        all_regions = lh_cortical + rh_cortical + subcortical

        # Pad to 112 if needed
        while len(all_regions) < 112:
            all_regions.append(f"Region_{len(all_regions)}")

        return all_regions[:112]

    def _define_ad_pathology_regions(self) -> Dict[str, List[str]]:
        """Define regions known to be affected in AD"""
        return {
            'Braak_I_II': [  # Transentorhinal
                'L_Entorhinal', 'R_Entorhinal',
                'L_Parahippocampal', 'R_Parahippocampal'
            ],
            'Braak_III_IV': [  # Limbic
                'L_Hippocampus', 'R_Hippocampus',
                'L_Amygdala', 'R_Amygdala',
                'L_Fusiform', 'R_Fusiform',
                'L_MiddleTemporal', 'R_MiddleTemporal'
            ],
            'Braak_V_VI': [  # Isocortical
                'L_InferiorParietal', 'R_InferiorParietal',
                'L_SuperiorTemporal', 'R_SuperiorTemporal',
                'L_Precuneus', 'R_Precuneus'
            ],
            'DMN_hubs': [
                'L_PosteriorCingulate', 'R_PosteriorCingulate',
                'L_Precuneus', 'R_Precuneus',
                'L_InferiorParietal', 'R_InferiorParietal',
                'L_MedialOrbitofrontal', 'R_MedialOrbitofrontal'
            ],
            'Amyloid_early': [
                'L_Precuneus', 'R_Precuneus',
                'L_PosteriorCingulate', 'R_PosteriorCingulate'
            ]
        }

    def compute_node_importance_integrated_gradients(self, data_loader, n_steps=50):
        """
        Compute node importance using Integrated Gradients.
        Most stable gradient-based attribution method.
        """
        logger.info("Computing Integrated Gradients node importance...")

        all_node_attributions = []
        all_feature_attributions = []
        all_labels = []
        all_predictions = []

        for batch in data_loader:
            batch = batch.to(self.device)

            # Baseline (zeros)
            baseline = torch.zeros_like(batch.x)

            # Storage for this batch
            batch_node_attr = torch.zeros_like(batch.x)

            # Integrate gradients
            for step in range(n_steps + 1):
                alpha = step / n_steps
                interpolated_x = baseline + alpha * (batch.x - baseline)

                batch_copy = batch.clone()
                batch_copy.x = interpolated_x
                batch_copy.x.requires_grad = True

                # Forward pass
                output = self._forward(batch_copy)

                # Gradient of AD class probability
                ad_score = output[:, 1].sum()

                if ad_score.requires_grad:
                    grads = torch.autograd.grad(
                        outputs=ad_score,
                        inputs=batch_copy.x,
                        create_graph=False,
                        retain_graph=False
                    )[0]

                    batch_node_attr = batch_node_attr + grads.to(batch_node_attr.dtype) / float(n_steps)

            # Multiply by (input - baseline)
            attributions = batch_node_attr * (batch.x - baseline)

            # Store per-graph results
            batch_ids = batch.batch.cpu().numpy()
            for graph_id in np.unique(batch_ids):
                mask = batch_ids == graph_id

                # Node importance (sum absolute attribution across features)
                node_imp = attributions[mask].abs().sum(dim=1).detach().cpu().numpy()
                all_node_attributions.append(node_imp)

                # Feature importance (sum across nodes)
                feat_imp = attributions[mask].abs().sum(dim=0).detach().cpu().numpy()
                all_feature_attributions.append(feat_imp)

                # Per-graph prediction & label
                with torch.no_grad():
                    out = self._forward(batch)
                    pred = out.argmax(dim=1)[graph_id].item()
                all_predictions.append(pred)

                label = batch.y[graph_id].item()
                all_labels.append(label)


                # Get prediction for this batch
                with torch.no_grad():
                    if hasattr(batch, 'batch'):
                        # For batched graphs
                        pred_logits = self._forward(batch)
                        if len(pred_logits) > 0:
                            pred = pred_logits.argmax(dim=1)[0].item()
                        else:
                            pred = 0  # Default prediction
                    else:
                        # For single graph
                        pred_logits = self._forward(batch)
                        pred = pred_logits.argmax(dim=1)[0].item()

        # Convert to arrays
        node_attributions = np.array(all_node_attributions)  # [n_samples, n_nodes]
        feature_attributions = np.array(all_feature_attributions)  # [n_samples, n_features]

        results = {
            'node_importance_mean': node_attributions.mean(axis=0),
            'node_importance_std': node_attributions.std(axis=0),
            'node_importance_per_sample': node_attributions,
            'feature_importance_mean': feature_attributions.mean(axis=0),
            'feature_importance_std': feature_attributions.std(axis=0),
            'feature_importance_per_sample': feature_attributions,
            'labels': np.array(all_labels),
            'predictions': np.array(all_predictions)
        }

        self.importance_cache['integrated_gradients'] = results
        return results

    def compute_feature_decomposition(self, data_loader):
        """
        Decompose importance by feature type. Robust to presence/absence of
        batch.feature_idx_map and to ablation setups with fewer columns.

        Groups (original index space):
        - Connectivity:    [0, 1, 2]
        - Demographics:    [3, 4]
        - APOE:            [5, 6, 7, 8, 9, 10]
        - TAU_PET:         [11, 15]
        - Amyloid_PET:     [12, 16]
        - sMRI:            [13, 14, 17, 18]
        """
        logger.info("Computing feature-specific importance decomposition...")

        # Original-index groups (do not change these)
        feature_groups = {
            'Connectivity': [0, 1, 2],
            'Demographics': [3, 4],
            'APOE': [5, 6, 7, 8, 9, 10],
            'TAU_PET': [11, 15],
            'Amyloid_PET': [12, 16],
            'sMRI': [13, 14, 17, 18],
        }

        from collections import defaultdict
        group_importance = defaultdict(list)

        for batch in data_loader:
            batch = batch.to(self.device)
            batch.x.requires_grad = True

            output = self._forward(batch)
            ad_score = output[:, 1].sum()

            if not ad_score.requires_grad:
                continue

            grads = torch.autograd.grad(
                outputs=ad_score,
                inputs=batch.x,
                create_graph=False
            )[0]  # [num_nodes_in_batch, num_features_current]

            # ---- Build/normalize feature_idx_map -> tensor of length == current feature width ----
            feat_map = getattr(batch, 'feature_idx_map', None)

            # If absent: identity map (current col t corresponds to original index t).
            if feat_map is None:
                feat_map = torch.arange(grads.shape[1], device=grads.device)
            else:
                # Convert to tensor if it's list/np
                if isinstance(feat_map, (list, tuple)):
                    feat_map = torch.tensor(feat_map, device=grads.device)
                elif isinstance(feat_map, np.ndarray):
                    feat_map = torch.from_numpy(feat_map).to(grads.device)
                # If PyG Batch concatenation changed widths, trim to current dimension
                if feat_map.numel() != grads.shape[1]:
                    feat_map = feat_map[:grads.shape[1]]

            # Set of "original indices" that actually exist in this batch
            available_orig = set(feat_map.tolist())

            # ---- For each feature group, find CURRENT columns that correspond to the group's ORIGINAL indices ----
            for group_name, orig_indices in feature_groups.items():
                current_cols = []
                for k in orig_indices:
                    if k not in available_orig:
                        continue
                    # position t where current column maps back to original index k
                    hits = (feat_map == k).nonzero(as_tuple=True)[0]
                    if hits.numel() > 0:
                        t = int(hits[0].item())
                        if 0 <= t < grads.shape[1]:
                            current_cols.append(t)

                if len(current_cols) == 0:
                    # Group absent in this ablation or fully dropped: contribute zero (no warnings)
                    group_importance[group_name].append(0.0)
                    continue

                # Safe gather and aggregate absolute gradients for the group
                group_grads = grads[:, current_cols].abs().sum()
                group_importance[group_name].append(group_grads.item())

        # Average across batches
        results = {
            group: {
                'mean': float(np.mean(values)) if len(values) else 0.0,
                'std': float(np.std(values)) if len(values) else 0.0,
                'values': values
            }
            for group, values in group_importance.items()
        }

        self.importance_cache['feature_decomposition'] = results
        return results

    def extract_attention_weights(self, data_loader):
        """
        Extract attention weights from GAT model.
        Only works if model has attention mechanism.
        """
        if not hasattr(self.model, 'convs'):
            logger.warning("Model doesn't have attention layers")
            return None

        logger.info("Extracting attention weights from GAT layers...")

        attention_maps = []

        for batch in data_loader:
            batch = batch.to(self.device)

            # Hook to capture attention weights
            attention_weights = []

            def attention_hook(module, input, output):
                # GAT layers return (x, (edge_index, attention_weights))
                if isinstance(output, tuple) and len(output) == 2:
                    _, attn_tuple = output
                    if isinstance(attn_tuple, tuple):
                        attention_weights.append(attn_tuple[1].detach().cpu())

            # Temporarily turn on attention returns (if supported), and hook outputs
            hooks = []
            old_flags = []
            for conv in self.model.convs:
                # Save & set flag if exists
                if hasattr(conv, 'return_attention_weights'):
                    old_flags.append((conv, conv.return_attention_weights))
                    conv.return_attention_weights = True
                else:
                    old_flags.append((conv, None))
                # Hook to capture outputs (x, (edge_index, alpha)) tuples
                hooks.append(conv.register_forward_hook(attention_hook))

            # Forward pass
            with torch.no_grad():
                _ = self._forward(batch)

            # Remove hooks and restore flags
            for hook in hooks:
                hook.remove()
            for conv, old in old_flags:
                if old is not None:
                    conv.return_attention_weights = old


            if attention_weights:
                attention_maps.append(attention_weights)

        return attention_maps

    def compute_class_specific_importance(self, data_loader):
        """
        Compute separate importance maps for CN and AD samples.
        Identifies regions that discriminate in each direction.
        """
        logger.info("Computing class-specific importance...")

        cn_attributions = []
        ad_attributions = []

        for batch in data_loader:
            batch = batch.to(self.device)
            batch.x.requires_grad = True

            output = self._forward(batch)

            # Separate by true label (work graph-by-graph)
            batch_ids = batch.batch.cpu().numpy()              # length = num_nodes_in_batch
            unique_graphs = np.unique(batch_ids)

            for graph_id in unique_graphs:
                # node mask for this graph (numpy -> torch.bool on the same device)
                node_mask_np = (batch_ids == graph_id)
                node_mask = torch.from_numpy(node_mask_np).to(batch.x.device).bool()

                # graph-level label is indexed by graph_id (NOT by node mask)
                label = int(batch.y[graph_id].item())

                # Score of the *true* class for this graph
                score = output[graph_id, label]

                if score.requires_grad:
                    # gradient wrt all node features in the batch (keep graph for next iterations)
                    grads = torch.autograd.grad(
                        outputs=score,
                        inputs=batch.x,
                        retain_graph=True
                    )[0]

                    # restrict to nodes that belong to this graph
                    node_attr = grads[node_mask].abs().sum(dim=1).detach().cpu().numpy()

                    if label == 0:  # CN
                        cn_attributions.append(node_attr)
                    else:           # AD
                        ad_attributions.append(node_attr)


        results = {
            'cn_importance': np.array(cn_attributions).mean(axis=0) if cn_attributions else np.zeros(112),
            'ad_importance': np.array(ad_attributions).mean(axis=0) if ad_attributions else np.zeros(112),
            'differential': None
        }

        if cn_attributions and ad_attributions:
            results['differential'] = results['ad_importance'] - results['cn_importance']

        return results

    def compare_to_pathology(self, node_importance):
        """
        Compare learned importance to known AD pathology patterns.
        Returns overlap statistics and rankings.
        """
        logger.info("Comparing to AD pathology patterns...")

        results = {}

        # Compute importance for each pathology stage
        for stage, region_list in self.ad_pathology_regions.items():
            stage_importance = []

            for region in region_list:
                if region in self.region_names:
                    idx = self.region_names.index(region)
                    if idx < len(node_importance):
                        stage_importance.append(float(node_importance[idx]))

            if stage_importance:
                results[stage] = {
                    'mean_importance': np.mean(stage_importance),
                    'std_importance': np.std(stage_importance),
                    'regions': region_list,
                    'values': stage_importance
                }

        # Top regions
        top_k = 20
        top_indices = np.argsort(node_importance)[-top_k:][::-1]
        top_regions = [
            {
                'rank': i+1,
                'region': self.region_names[idx],
                'importance': float(node_importance[idx])
            }
            for i, idx in enumerate(top_indices)
        ]

        # Overlap with pathology
        all_pathology_regions = set()
        for regions in self.ad_pathology_regions.values():
            all_pathology_regions.update(regions)

        top_region_names = [r['region'] for r in top_regions]
        overlap_count = len(set(top_region_names) & all_pathology_regions)

        results['top_regions'] = top_regions
        results['overlap_with_pathology'] = {
            'count': overlap_count,
            'percentage': 100 * overlap_count / min(top_k, len(all_pathology_regions)),
            'overlapping_regions': list(set(top_region_names) & all_pathology_regions)
        }

        return results

    def save_model_weights(self, output_path: Path):
        """
        Save complete model weights and architecture info.
        """
        logger.info(f"Saving model weights to {output_path}")

        # Save full model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
        }, output_path / 'model_weights.pth')

        # Extract and save layer-wise weights
        layer_weights = {}
        for name, param in self.model.named_parameters():
            layer_weights[name] = param.detach().cpu().numpy()

        with open(output_path / 'layer_weights.pkl', 'wb') as f:
            pickle.dump(layer_weights, f)

        # Save weight statistics
        weight_stats = {}
        for name, param in self.model.named_parameters():
            weight_stats[name] = {
                'shape': list(param.shape),
                'mean': float(param.mean()),
                'std': float(param.std()),
                'min': float(param.min()),
                'max': float(param.max()),
                'num_params': int(param.numel())
            }

        with open(output_path / 'weight_statistics.json', 'w') as f:
            json.dump(weight_stats, f, indent=2)

        logger.info("Model weights saved successfully")

    def save_importance_results(self, output_path: Path):
        """
        Save all importance results for external analysis.
        """
        logger.info(f"Saving importance results to {output_path}")

        # Save main results as pickle (includes arrays)
        with open(output_path / 'importance_results.pkl', 'wb') as f:
            pickle.dump(self.importance_cache, f)

        # Save JSON version (for readability, excluding large arrays)
        json_results = {}
        for key, value in self.importance_cache.items():
            if isinstance(value, dict):
                json_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) and v.size < 1000 else
                       (f"<array shape={v.shape}>" if isinstance(v, np.ndarray) else v)
                    for k, v in value.items()
                }

        with open(output_path / 'importance_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info("Importance results saved successfully")

    def create_comprehensive_visualization(self, output_path: Path):
        """
        Create comprehensive multi-panel visualization.
        """
        logger.info("Creating comprehensive visualization...")

        if 'integrated_gradients' not in self.importance_cache:
            logger.warning("Run compute_node_importance_integrated_gradients first")
            return

        ig_results = self.importance_cache['integrated_gradients']
        node_imp = ig_results['node_importance_mean']
        feat_imp = ig_results['feature_importance_mean']

        # Create figure
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. top 30 brain regions
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        top_k = 30
        top_indices = np.argsort(node_imp)[-top_k:]
        top_regions = [self.region_names[i] for i in top_indices]
        top_values = node_imp[top_indices]

        # Color by pathology
        all_pathology = set()
        for regions in self.ad_pathology_regions.values():
            all_pathology.update(regions)

        colors = ['#d62728' if r in all_pathology else '#1f77b4' for r in top_regions]

        y_pos = np.arange(len(top_regions))
        ax1.barh(y_pos, top_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_regions, fontsize=8)
        ax1.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax1.set_title(f'Top {top_k} Most Important Brain Regions\n(Red = Known AD pathology)',
                     fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # 2. feature group importance
        ax2 = fig.add_subplot(gs[0, 2])
        if 'feature_decomposition' in self.importance_cache:
            feat_decomp = self.importance_cache['feature_decomposition']
            groups = list(feat_decomp.keys())
            values = [feat_decomp[g]['mean'] for g in groups]
            errors = [feat_decomp[g]['std'] for g in groups]

            ax2.barh(groups, values, xerr=errors, alpha=0.7, color='#2ca02c',
                    edgecolor='black', capsize=5)
            ax2.set_xlabel('Mean Importance', fontsize=10, fontweight='bold')
            ax2.set_title('Feature Group Importance', fontsize=11, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)

        # 3. braak stage comparison
        ax3 = fig.add_subplot(gs[0, 3])
        pathology_comp = self.compare_to_pathology(node_imp)

        stages = []
        stage_values = []
        for stage in ['Braak_I_II', 'Braak_III_IV', 'Braak_V_VI']:
            if stage in pathology_comp:
                stages.append(stage.replace('Braak_', 'Stage ').replace('_', '-'))
                stage_values.append(pathology_comp[stage]['mean_importance'])

        if stages:
            ax3.bar(range(len(stages)), stage_values, alpha=0.7,
                   color=['#ff7f0e', '#ff7f0e', '#ff7f0e'], edgecolor='black')
            ax3.set_xticks(range(len(stages)))
            ax3.set_xticklabels(stages, rotation=45, ha='right')
            ax3.set_ylabel('Mean Importance', fontsize=10, fontweight='bold')
            ax3.set_title('Braak Stage Importance', fontsize=11, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)

        # 4. node importance heatmap
        ax4 = fig.add_subplot(gs[1, 2:])
        # Reshape importance to matrix (56x2 for hemispheres)
        n_cortical = 68  # Cortical regions
        heatmap_data = np.zeros((2, n_cortical//2))
        for i in range(min(n_cortical//2, len(node_imp))):
            heatmap_data[0, i] = node_imp[i]  # Left
        for i in range(min(n_cortical//2, len(node_imp) - n_cortical//2)):
            heatmap_data[1, i] = node_imp[i + n_cortical//2]  # Right

        im = ax4.imshow(heatmap_data, cmap='hot', aspect='auto')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Left Hemisphere', 'Right Hemisphere'])
        ax4.set_xlabel('Cortical Region Index', fontsize=10, fontweight='bold')
        ax4.set_title('Hemispheric Importance Map', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax4, label='Importance')

        # 5. distribution
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(node_imp, bins=40, edgecolor='black', alpha=0.7, color='#9467bd')
        ax5.axvline(node_imp.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {node_imp.mean():.3f}')
        ax5.set_xlabel('Importance Score', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax5.set_title('Importance Distribution', fontsize=11, fontweight='bold')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)

        # 6. network-wise
        ax6 = fig.add_subplot(gs[2, 1])
        network_imp = self._aggregate_by_network(node_imp)
        if network_imp:
            networks = list(network_imp.keys())
            values = list(network_imp.values())
            ax6.bar(networks, values, alpha=0.7, color='#8c564b', edgecolor='black')
            ax6.set_ylabel('Mean Importance', fontsize=10, fontweight='bold')
            ax6.set_title('Network-wise Importance', fontsize=11, fontweight='bold')
            ax6.tick_params(axis='x', rotation=45, labelsize=8)
            ax6.grid(axis='y', alpha=0.3)

        # 7. prediction performance
        ax7 = fig.add_subplot(gs[2, 2])
        labels = ig_results['labels']
        predictions = ig_results['predictions']

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7,
                   xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'])
        ax7.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax7.set_ylabel('True', fontsize=10, fontweight='bold')
        ax7.set_title('Confusion Matrix', fontsize=11, fontweight='bold')

        # 8. top region table
        ax8 = fig.add_subplot(gs[2:, 3])
        ax8.axis('off')

        top_10 = pathology_comp['top_regions'][:10]
        table_data = [[r['rank'], r['region'][:20], f"{r['importance']:.4f}"]
                     for r in top_10]

        table = ax8.table(cellText=table_data,
                         colLabels=['Rank', 'Region', 'Importance'],
                         cellLoc='left', loc='center',
                         colWidths=[0.15, 0.55, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        ax8.set_title('Top 10 Regions (Detailed)', fontsize=11, fontweight='bold', pad=20)

        # 9. subcortical importance
        ax9 = fig.add_subplot(gs[3, 0:2])
        subcort_start = 68
        subcort_regions = self.region_names[subcort_start:min(subcort_start+20, len(self.region_names))]
        subcort_imp = node_imp[subcort_start:min(subcort_start+20, len(node_imp))]

        if len(subcort_imp) > 0:
            ax9.barh(range(len(subcort_regions)), subcort_imp,
                    alpha=0.7, color='#e377c2', edgecolor='black')
            ax9.set_yticks(range(len(subcort_regions)))
            ax9.set_yticklabels(subcort_regions, fontsize=8)
            ax9.set_xlabel('Importance Score', fontsize=10, fontweight='bold')
            ax9.set_title('Subcortical Structure Importance', fontsize=11, fontweight='bold')
            ax9.grid(axis='x', alpha=0.3)

        # 10. feature importance detail
        ax10 = fig.add_subplot(gs[3, 2])
        feature_names = [
            'Strength', 'Degree', 'Clustering',
            'Age', 'Sex',
            'APOE2_0', 'APOE2_1', 'APOE2_2', 'APOE4_0', 'APOE4_1', 'APOE4_2',
            'TAU', 'Amyloid', 'Volume', 'Thickness',
            'TAU+', 'Amy+', 'Vol-', 'Thick-'
        ]

        if len(feat_imp) == len(feature_names):
            top_feat_idx = np.argsort(feat_imp)[-10:]
            ax10.barh(range(len(top_feat_idx)), feat_imp[top_feat_idx],
                     alpha=0.7, color='#bcbd22', edgecolor='black')
            ax10.set_yticks(range(len(top_feat_idx)))
            ax10.set_yticklabels([feature_names[i] for i in top_feat_idx], fontsize=8)
            ax10.set_xlabel('Importance', fontsize=10, fontweight='bold')
            ax10.set_title('Top 10 Individual Features', fontsize=11, fontweight='bold')
            ax10.grid(axis='x', alpha=0.3)

        plt.suptitle('Comprehensive GNN Interpretability Analysis',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.savefig(output_path / 'comprehensive_interpretability.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualization saved to {output_path}")

    def _aggregate_by_network(self, node_importance):
        """Aggregate by functional network"""
        networks = {
            'MTL': ['Hippocampus', 'Entorhinal', 'Parahippocampal', 'Amygdala'],
            'DMN': ['Precuneus', 'PosteriorCingulate', 'InferiorParietal'],
            'Temporal': ['Temporal', 'Fusiform'],
            'Frontal': ['Frontal', 'Cingulate'],
            'Parietal': ['Parietal', 'Supramarginal'],
            'Occipital': ['Occipital', 'Cuneus', 'Lingual']
        }

        network_imp = {}
        for net_name, keywords in networks.items():
            importances = []
            for i, region in enumerate(self.region_names[:len(node_importance)]):
                if any(kw in region for kw in keywords):
                    importances.append(node_importance[i])

            if importances:
                network_imp[net_name] = float(np.mean(importances))

        return network_imp


def run_complete_interpretability_pipeline(model, test_loader, output_dir: Path,
                                          experiment_name: str, device='cuda'):
    """
    Run complete interpretability analysis and save all results.

    Args:
        model: Trained GNN model
        test_loader: Test data loader
        output_dir: Output directory
        experiment_name: Name of experiment
        device: torch device
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"INTERPRETABILITY ANALYSIS: {experiment_name}")
    logger.info(f"{'='*80}\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = EnhancedGNNInterpretability(model, device)

    # 1. Node importance (Integrated Gradients)
    ig_results = analyzer.compute_node_importance_integrated_gradients(test_loader, n_steps=50)

    # 2. Feature decomposition
    feat_decomp = analyzer.compute_feature_decomposition(test_loader)

    # 3. Class-specific importance
    class_specific = analyzer.compute_class_specific_importance(test_loader)

    # 4. Compare to pathology
    pathology_comp = analyzer.compare_to_pathology(ig_results['node_importance_mean'])

    # 5. Save model weights
    analyzer.save_model_weights(output_dir)

    # 6. Save importance results
    analyzer.save_importance_results(output_dir)

    # 7. Create visualization
    analyzer.create_comprehensive_visualization(output_dir)

    # 8. Save summary report
    summary = {
        'experiment_name': experiment_name,
        'top_20_regions': pathology_comp['top_regions'],
        'pathology_overlap': pathology_comp['overlap_with_pathology'],
        'feature_importance': feat_decomp,
        'class_specific_differential': class_specific.get('differential', []).tolist() if class_specific.get('differential') is not None else None
    }

    with open(output_dir / 'interpretability_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("INTERPRETABILITY ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*80}\n")

    return analyzer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Enhanced interpretability module loaded")