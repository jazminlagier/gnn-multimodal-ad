#!/usr/bin/env python3
"""
Data loading and graph construction utilities.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
import logging
import random
from typing import Optional, List
from copy import deepcopy

logger = logging.getLogger(__name__)


class ADNIDataLoader:
    def __init__(self, fc_matrices_dir, demographics_file, augmentation_config=None):
        self.fc_matrices_dir = Path(fc_matrices_dir)
        self.demographics_file = Path(demographics_file)
        self.demographics_df = None

        # Initialize augmentor if config provided
        self.augmentor = None
        if augmentation_config and augmentation_config.get('enabled', False):
            augmentor_params = {
                'edge_noise_std': augmentation_config.get('edge_noise_std', 0.05),
                'node_mask_prob': augmentation_config.get('node_mask_prob', 0.15),
                'feature_scale_range': augmentation_config.get('feature_scale_range', (0.9, 1.1)),
                'network_permute_prob': augmentation_config.get('network_permute_prob', None),
            }
            self.augmentor = GraphAugmentor(**augmentor_params)
            self.target_ratio = augmentation_config.get('target_ratio', 0.4)
        else:
            self.target_ratio = 0.4

    def load_demographics(self):
        logger.info(f"Loading demographics from {self.demographics_file}")
        if not self.demographics_file.exists():
            raise FileNotFoundError(f"Demographics file not found: {self.demographics_file}")
        self.demographics_df = pd.read_csv(self.demographics_file)
        logger.info(f"Loaded demographics for {len(self.demographics_df)} entries")
        logger.info(f"Available columns: {list(self.demographics_df.columns)}")

        if 'diagnosis' in self.demographics_df.columns:
            dist = self.demographics_df['diagnosis'].value_counts()
            logger.info(f"Diagnosis distribution: {dist.to_dict()}")
        else:
            logger.warning("No 'diagnosis' column found in demographics data!")
        return self.demographics_df

    def augment_minority_class(self, graphs, labels, target_ratio=None):
        """Augment minority class (AD) graphs"""
        if self.augmentor is None:
            return graphs, labels

        if target_ratio is None:
            target_ratio = getattr(self, 'target_ratio', 0.4)

        labels_array = np.array(labels)
        cn_count = np.sum(labels_array == 0)
        ad_count = np.sum(labels_array == 1)

        total_needed = int(cn_count / (1 - target_ratio))
        ad_needed = max(0, total_needed - cn_count - ad_count)

        if ad_needed == 0:
            return graphs, labels

        logger.info(f"Augmenting AD class: current={ad_count}, needed={ad_needed}")

        ad_graphs = [graphs[i] for i in range(len(graphs)) if labels[i] == 1]
        augmented_graphs = list(graphs)
        augmented_labels = list(labels)

        for _ in range(ad_needed):
            base_graph = random.choice(ad_graphs)
            aug_graph = self.augmentor.augment_graph(base_graph, ['edge_jitter'])
            augmented_graphs.append(aug_graph)
            augmented_labels.append(1)

        return augmented_graphs, augmented_labels

    def load_fc_matrices(self):
        """Load Desikan-Killiany FC matrices (112x112)"""
        logger.info("Loading Desikan-Killiany FC matrices...")
        if self.demographics_df is None:
            raise ValueError("Demographics not loaded. Call load_demographics() first.")

        fc_matrices = []
        metadata = []
        num_missing_fc = 0
        num_bad_metadata = 0

        # DK directory
        dk_dir = self.fc_matrices_dir

        for _, row in self.demographics_df.iterrows():
            participant_id = row['participant_id']
            session_id = row.get('session_id', row.get('sess_id'))

            # Look for .netcc file
            fc_dir = dk_dir / participant_id / session_id
            fc_file = fc_dir / f"{participant_id}_{session_id}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-desikanKilliany_000.netcc"

            if not fc_file.exists():
                num_missing_fc += 1
                continue

            matrix = self._load_dk_matrix(fc_file)
            if matrix is None or matrix.shape != (112, 112):
                num_missing_fc += 1
                continue

            demo_data = self._extract_demo(row)
            if demo_data['age'] <= 0 or demo_data['sex'] not in ['M', 'F'] \
            or demo_data['diagnosis'] not in ['CN', 'AD']:
                num_bad_metadata += 1
                continue

            # Extract RID from participant_id
            rid = None
            if participant_id.startswith('sub-'):
                pid_clean = participant_id[4:]  # Remove 'sub-'
                if 'S' in pid_clean:
                    rid_str = pid_clean.split('S')[1]
                    try:
                        rid = int(rid_str)
                    except:
                        pass

            fc_matrices.append(matrix)
            metadata.append({
                'subject_id': participant_id,
                'participant_id': participant_id,
                'session_id': session_id,
                'rid': rid,
                'matrix_index': len(fc_matrices) - 1,
                'file_path': str(fc_file),
                **demo_data
            })

        logger.info(f"Successfully loaded {len(fc_matrices)} FC matrices")
        logger.info(f"Skipped due to missing FC: {num_missing_fc}")
        logger.info(f"Skipped due to bad metadata: {num_bad_metadata}")

        if not fc_matrices:
            raise ValueError("No valid FC matrices found!")

        fc_matrices = np.array(fc_matrices)
        metadata_df = pd.DataFrame(metadata)
        logger.info(f"Final dataset: {fc_matrices.shape}")
        logger.info(f"RID availability: {metadata_df['rid'].notna().sum()}/{len(metadata_df)}")
        return fc_matrices, metadata_df

    def _load_dk_matrix(self, fc_file):
        """Load DK FC matrix from .netcc file (handles variable sizes)"""
        try:
            with open(fc_file, 'r') as f:
                lines = f.readlines()

            # Find FZ section
            fz_start = None
            for i, line in enumerate(lines):
                if line.strip() == '# FZ':
                    fz_start = i + 1
                    break

            if fz_start is None:
                return None

            # Read matrix lines
            matrix_lines = []
            for i in range(fz_start, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('#'):
                    break
                matrix_lines.append(line)

            if len(matrix_lines) == 0:
                return None

            # Parse matrix
            matrix = []
            for line in matrix_lines:
                values = line.split('\t')
                row = [float(v.strip()) for v in values if v.strip()]
                matrix.append(row)

            matrix = np.array(matrix, dtype=np.float32)

            # Get actual size
            actual_size = matrix.shape[0]

            # Pad or trim to 112x112
            if actual_size != 112:
                logger.warning(f"DK matrix size {actual_size}x{actual_size}, adjusting to 112x112")
                new_matrix = np.zeros((112, 112), dtype=np.float32)
                copy_size = min(actual_size, 112)
                new_matrix[:copy_size, :copy_size] = matrix[:copy_size, :copy_size]
                matrix = new_matrix

            # Validate
            if np.isnan(matrix).any() or np.isinf(matrix).any():
                return None

            return matrix

        except Exception as e:
            logger.warning(f"Error loading DK matrix: {str(e)}")
            return None

    def _extract_demo(self, row):
        """Extract and normalize demographic information + biomarkers"""
        # Age
        age = 0.0
        for age_col in ['age', 'Age', 'demo_age']:
            if age_col in row and pd.notna(row[age_col]):
                try:
                    age = float(row[age_col])
                    break
                except (ValueError, TypeError):
                    continue

        # Sex
        sex = 'Unknown'
        for sex_col in ['sex', 'Sex', 'gender', 'demo_sex']:
            if sex_col in row and pd.notna(row[sex_col]):
                val = str(row[sex_col]).strip().upper()
                if val in ['M', 'MALE']:
                    sex = 'M'
                    break
                if val in ['F', 'FEMALE']:
                    sex = 'F'
                    break

        # Diagnosis
        diagnosis = 'Unknown'
        if 'diagnosis' in row and pd.notna(row['diagnosis']):
            raw = str(row['diagnosis']).strip().upper()
            if raw in ['CN', 'AD']:
                diagnosis = raw

        # APOE
        apoe4_count = 0
        apoe2_count = 0
        apoe_genotype = ''
        apoe_risk = 0

        if 'genotype' in row and pd.notna(row['genotype']):
            apoe_str = str(row['genotype']).strip()
            if '/' in apoe_str:
                alleles = apoe_str.split('/')
                if len(alleles) == 2:
                    try:
                        allele1 = int(alleles[0].strip())
                        allele2 = int(alleles[1].strip())
                        apoe4_count = sum(1 for allele in (allele1, allele2) if allele == 4)
                        apoe2_count = sum(1 for allele in (allele1, allele2) if allele == 2)

                        a_sorted = tuple(sorted([allele1, allele2]))
                        apoe_genotype = f"{a_sorted[0]}/{a_sorted[1]}"
                        apoe_risk = 1 if (4 in a_sorted) else 0
                    except (ValueError, TypeError):
                        pass

        # Tau biomarkers
        tau_meta_temporal = 0.0
        tau_entorhinal = 0.0
        if 'META_TEMPORAL_SUVR' in row and pd.notna(row['META_TEMPORAL_SUVR']):
            tau_meta_temporal = float(row['META_TEMPORAL_SUVR'])
        if 'CTX_ENTORHINAL_SUVR' in row and pd.notna(row['CTX_ENTORHINAL_SUVR']):
            tau_entorhinal = float(row['CTX_ENTORHINAL_SUVR'])

        # Amyloid biomarkers
        amyloid_status = 0.0
        centiloids = 0.0
        if 'AMYLOID_STATUS' in row and pd.notna(row['AMYLOID_STATUS']):
            amyloid_status = float(row['AMYLOID_STATUS'])
        if 'CENTILOIDS' in row and pd.notna(row['CENTILOIDS']):
            centiloids = float(row['CENTILOIDS'])

        # Binary flags
        tau_positive = 1.0 if tau_meta_temporal > 1.3 else 0.0
        amyloid_positive = amyloid_status

        # Extract sMRI features
        smri_hippocampus = 0.0
        smri_entorhinal = 0.0
        smri_amygdala = 0.0
        smri_gray_matter = 0.0
        smri_ventricles = 0.0
        smri_temporal = 0.0

        if 'SMRI_HIPPOCAMPUS_BILATERAL' in row and pd.notna(row['SMRI_HIPPOCAMPUS_BILATERAL']):
            smri_hippocampus = float(row['SMRI_HIPPOCAMPUS_BILATERAL'])
        if 'SMRI_ENTORHINAL_BILATERAL' in row and pd.notna(row['SMRI_ENTORHINAL_BILATERAL']):
            smri_entorhinal = float(row['SMRI_ENTORHINAL_BILATERAL'])
        if 'SMRI_AMYGDALA_BILATERAL' in row and pd.notna(row['SMRI_AMYGDALA_BILATERAL']):
            smri_amygdala = float(row['SMRI_AMYGDALA_BILATERAL'])
        if 'SMRI_TOTAL_GRAY_MATTER' in row and pd.notna(row['SMRI_TOTAL_GRAY_MATTER']):
            smri_gray_matter = float(row['SMRI_TOTAL_GRAY_MATTER'])
        if 'SMRI_VENTRICLES' in row and pd.notna(row['SMRI_VENTRICLES']):
            smri_ventricles = float(row['SMRI_VENTRICLES'])
        if 'SMRI_TEMPORAL_COMPOSITE' in row and pd.notna(row['SMRI_TEMPORAL_COMPOSITE']):
            smri_temporal = float(row['SMRI_TEMPORAL_COMPOSITE'])

        # Binary sMRI indicators
        smri_hippocampal_atrophy = 0.0
        smri_entorhinal_thinning = 0.0
        smri_ventricular_enlargement = 0.0
        smri_ad_pattern = 0.0

        if 'SMRI_HIPPOCAMPAL_ATROPHY' in row and pd.notna(row['SMRI_HIPPOCAMPAL_ATROPHY']):
            smri_hippocampal_atrophy = float(row['SMRI_HIPPOCAMPAL_ATROPHY'])
        if 'SMRI_ENTORHINAL_THINNING' in row and pd.notna(row['SMRI_ENTORHINAL_THINNING']):
            smri_entorhinal_thinning = float(row['SMRI_ENTORHINAL_THINNING'])
        if 'SMRI_VENTRICULAR_ENLARGEMENT' in row and pd.notna(row['SMRI_VENTRICULAR_ENLARGEMENT']):
            smri_ventricular_enlargement = float(row['SMRI_VENTRICULAR_ENLARGEMENT'])
        if 'SMRI_AD_PATTERN_POSITIVE' in row and pd.notna(row['SMRI_AD_PATTERN_POSITIVE']):
            smri_ad_pattern = float(row['SMRI_AD_PATTERN_POSITIVE'])

        return {
            'age': age,
            'sex': sex,
            'diagnosis': diagnosis,
            'apoe4_count': apoe4_count,
            'apoe2_count': apoe2_count,
            'apoe_genotype': apoe_genotype,
            'apoe_risk': apoe_risk,
            'tau_meta_temporal': tau_meta_temporal,
            'tau_entorhinal': tau_entorhinal,
            'tau_positive': tau_positive,
            'amyloid_status': amyloid_status,
            'amyloid_positive': amyloid_positive,
            'centiloids': centiloids,
            'smri_hippocampus': smri_hippocampus,
            'smri_entorhinal': smri_entorhinal,
            'smri_amygdala': smri_amygdala,
            'smri_gray_matter': smri_gray_matter,
            'smri_ventricles': smri_ventricles,
            'smri_temporal': smri_temporal,
            'smri_hippocampal_atrophy': smri_hippocampal_atrophy,
            'smri_entorhinal_thinning': smri_entorhinal_thinning,
            'smri_ventricular_enlargement': smri_ventricular_enlargement,
            'smri_ad_pattern': smri_ad_pattern
        }


class GraphConstructor:
    """Turn an FC matrix into a PyG Data graph"""
    def __init__(self, threshold=0.1, top_k=None):
        self.threshold = threshold
        self.top_k = top_k

    def construct_graph(self, fc_matrix, node_features):
        num_nodes = fc_matrix.shape[0]
        if fc_matrix.shape != (num_nodes, num_nodes):
            raise ValueError(f"Expected square matrix, got {fc_matrix.shape}")
        if node_features.shape[0] != num_nodes:
            raise ValueError(f"Node features shape {node_features.shape} doesn't match N={num_nodes}")

        # Upper triangle indices (exclude diagonal)
        edge_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
        fm = torch.tensor(fc_matrix, dtype=torch.float32)
        edge_weights = fm[edge_indices[0], edge_indices[1]]

        # Filter
        if self.top_k is not None:
            k = min(self.top_k, edge_weights.numel())
            _, keep = torch.topk(edge_weights.abs(), k)
            edge_indices = edge_indices[:, keep]
            edge_weights = edge_weights[keep]
        elif self.threshold is not None:
            mask = edge_weights.abs() > self.threshold
            edge_indices = edge_indices[:, mask]
            edge_weights = edge_weights[mask]

        # Undirected: mirror edges
        edge_index = torch.cat([edge_indices, edge_indices.flip(0)], dim=1)
        edge_attr = torch.cat([edge_weights, edge_weights], dim=0)

        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )


class GraphAugmentor:
    def __init__(self, edge_noise_std=0.05, node_mask_prob=0.15, feature_scale_range=(0.9, 1.1),
                 network_permute_prob=None, **kwargs):
        self.edge_noise_std = float(edge_noise_std)
        self.node_mask_prob = float(node_mask_prob)
        self.feature_scale_range = feature_scale_range
        self.network_permute_prob = network_permute_prob

    def augment_graph(self, graph, augment_types=None):
        """Apply edge-based augmentation to a PyG Data/Batch"""
        if augment_types is None:
            augment_types = ['edge_jitter']

        aug_graph = deepcopy(graph)

        with torch.no_grad():
            # Edge jitter
            if 'edge_jitter' in augment_types and getattr(aug_graph, 'edge_attr', None) is not None:
                noise = torch.empty_like(aug_graph.edge_attr).normal_(mean=0.0, std=self.edge_noise_std)
                aug_graph.edge_attr = torch.clamp(aug_graph.edge_attr + noise, -1.0, 1.0)

            # Node mask (first 3 connectivity features)
            if 'node_mask' in augment_types and getattr(aug_graph, 'x', None) is not None:
                N, F = aug_graph.x.shape
                if F >= 3:
                    mask = torch.rand(N, device=aug_graph.x.device) < self.node_mask_prob
                    x = aug_graph.x.clone()
                    x[mask, :3] = 0.0
                    aug_graph.x = x

            # Feature scale (first 3 features)
            if 'feature_scale' in augment_types and getattr(aug_graph, 'x', None) is not None:
                N, F = aug_graph.x.shape
                if F >= 3:
                    scale_min, scale_max = self.feature_scale_range
                    scale = torch.empty(N, 1, device=aug_graph.x.device, dtype=aug_graph.x.dtype).uniform_(scale_min, scale_max)
                    x = aug_graph.x.clone()
                    x[:, :3] = x[:, :3] * scale
                    aug_graph.x = x

        return aug_graph
