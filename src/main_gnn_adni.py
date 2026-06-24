#!/usr/bin/env python3
"""
Cross-validation runner for multimodal graph neural network experiments on DK-atlas brain graphs.

Runs participant-level 5-fold cross-validation across a set of feature ablations
defined in a YAML config. For each experiment it trains the selected models,
applies temperature scaling and threshold tuning on the validation fold, and
aggregates per-fold metrics and node-importance rankings.

Usage:
    python src/main_gnn_adni.py --config configs/example_config.yaml
"""

import sys
import json
import pickle
import random
import logging
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

sys.path.append(str(Path(__file__).parent))

from data_utils_v25 import ADNIDataLoader, GraphConstructor
from baseline_models_v25 import GCNBaseline, GATBaseline
from gkan_simple_v25 import SimpleGKAN
from regional_features_dk import DKRegionalFeatureExtractor, compute_node_features_dk_regional
from dk_interpretability import EnhancedGNNInterpretability
from learning_curve_tracker import LearningCurveTracker, GradientFlowAnalyzer
from train_utils_v25 import (
    train_model_with_tracking,
    evaluate_model_with_interpretability,
    train_model,
    train_model_optimized,
    evaluate_model,
    fit_temperature,
    tune_threshold_on_val,
)

# Optional FreeSurfer regional sMRI extractor. When the module and its input
# files are not present, the runner falls back to global sMRI features.
try:
    from regional_smri_freesurfer_extractor import FreeSurferRegionalExtractor
    REGIONAL_SMRI_AVAILABLE = True
except ImportError:
    REGIONAL_SMRI_AVAILABLE = False

# Populated in main() when regional sMRI is enabled; consumed by the
# node-feature helpers through this module-global reference.
freesurfer_extractor = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gnn_runner")


def load_config(config_path):
    """Load the YAML experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 5-fold cross-validation GNN experiments."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML configuration file (see configs/example_config.yaml).",
    )
    return parser.parse_args()


def compute_node_features(fc_tensor, age_z, sex_bin,
                          apoe2_0, apoe2_1, apoe2_2, apoe4_0, apoe4_1, apoe4_2,
                          tau_positive, amyloid_positive,
                          smri_hippocampus_z, smri_entorhinal_z, smri_amygdala_z,
                          smri_gray_matter_z, smri_ventricles_z, smri_temporal_z,
                          smri_hippocampal_atrophy, smri_entorhinal_thinning,
                          smri_ventricular_enlargement, smri_ad_pattern,
                          include_connectivity=True,
                          include_demographics=True,
                          include_apoe=True,
                          include_pet=True,
                          include_smri_continuous=True,
                          include_smri_binary=True,
                          include_age=True,
                          include_sex=True):
    N = fc_tensor.shape[0]
    feature_list = []

    with torch.no_grad():
        # Connectivity features
        if include_connectivity:
            W = fc_tensor.abs().clone()
            W.fill_diagonal_(0.0)
            strength = W.mean(dim=1)
            deg = W.sum(dim=1)
            dmin, dmax = deg.min(), deg.max()
            degree_norm = (deg - dmin) / (dmax - dmin + 1e-6)
            W13 = torch.pow(W, 1.0/3.0)
            T = W13 @ W13 @ W13
            k = (W > 0).sum(dim=1).float()
            denom = k * (k - 1.0)
            clust = torch.zeros(N, dtype=torch.float32, device=W.device)
            m = denom > 0
            clust[m] = torch.diag(T)[m] / denom[m]
            feature_list.extend([strength, degree_norm, clust])

        # Demographic features
        if include_demographics:
            if include_age:
                age_feat = torch.full((N,), float(age_z), dtype=torch.float32, device=fc_tensor.device)
                feature_list.append(age_feat)
            if include_sex:
                sex_feat = torch.full((N,), float(sex_bin), dtype=torch.float32, device=fc_tensor.device)
                feature_list.append(sex_feat)

        # APOE features
        if include_apoe:
            e2_0_f = torch.full((N,), float(apoe2_0), dtype=torch.float32, device=fc_tensor.device)
            e2_1_f = torch.full((N,), float(apoe2_1), dtype=torch.float32, device=fc_tensor.device)
            e2_2_f = torch.full((N,), float(apoe2_2), dtype=torch.float32, device=fc_tensor.device)
            e4_0_f = torch.full((N,), float(apoe4_0), dtype=torch.float32, device=fc_tensor.device)
            e4_1_f = torch.full((N,), float(apoe4_1), dtype=torch.float32, device=fc_tensor.device)
            e4_2_f = torch.full((N,), float(apoe4_2), dtype=torch.float32, device=fc_tensor.device)
            feature_list.extend([e2_0_f, e2_1_f, e2_2_f, e4_0_f, e4_1_f, e4_2_f])

        # PET biomarkers
        if include_pet:
            tau_feat = torch.full((N,), float(tau_positive), dtype=torch.float32, device=fc_tensor.device)
            amyloid_feat = torch.full((N,), float(amyloid_positive), dtype=torch.float32, device=fc_tensor.device)
            feature_list.extend([tau_feat, amyloid_feat])

        # sMRI continuous
        if include_smri_continuous:
            smri_hipp_f = torch.full((N,), float(smri_hippocampus_z), dtype=torch.float32, device=fc_tensor.device)
            smri_ento_f = torch.full((N,), float(smri_entorhinal_z), dtype=torch.float32, device=fc_tensor.device)
            smri_amyg_f = torch.full((N,), float(smri_amygdala_z), dtype=torch.float32, device=fc_tensor.device)
            smri_gray_f = torch.full((N,), float(smri_gray_matter_z), dtype=torch.float32, device=fc_tensor.device)
            smri_vent_f = torch.full((N,), float(smri_ventricles_z), dtype=torch.float32, device=fc_tensor.device)
            smri_temp_f = torch.full((N,), float(smri_temporal_z), dtype=torch.float32, device=fc_tensor.device)
            feature_list.extend([smri_hipp_f, smri_ento_f, smri_amyg_f, smri_gray_f, smri_vent_f, smri_temp_f])

        # sMRI binary
        if include_smri_binary:
            smri_hatrophy_f = torch.full((N,), float(smri_hippocampal_atrophy), dtype=torch.float32, device=fc_tensor.device)
            smri_ethin_f = torch.full((N,), float(smri_entorhinal_thinning), dtype=torch.float32, device=fc_tensor.device)
            smri_venlarge_f = torch.full((N,), float(smri_ventricular_enlargement), dtype=torch.float32, device=fc_tensor.device)
            smri_adpat_f = torch.full((N,), float(smri_ad_pattern), dtype=torch.float32, device=fc_tensor.device)
            feature_list.extend([smri_hatrophy_f, smri_ethin_f, smri_venlarge_f, smri_adpat_f])

        if len(feature_list) == 0:
            raise ValueError("At least one feature group must be enabled!")

        x = torch.stack(feature_list, dim=1)

    return x

# Node-feature construction using regional sMRI from FreeSurfer
def compute_node_features_enhanced_smri(fc_tensor, age_z, sex_bin,
                                       apoe2_0, apoe2_1, apoe2_2, apoe4_0, apoe4_1, apoe4_2,
                                       tau_positive, amyloid_positive,
                                       smri_hippocampus_z, smri_entorhinal_z, smri_amygdala_z,
                                       smri_gray_matter_z, smri_ventricles_z, smri_temporal_z,
                                       smri_hippocampal_atrophy, smri_entorhinal_thinning,
                                       smri_ventricular_enlargement, smri_ad_pattern,
                                       participant_id=None,  # required for regional sMRI lookup
                                       include_connectivity=True,
                                       include_demographics=True,
                                       include_apoe=True,
                                       include_pet=True,
                                       include_smri_continuous=True,
                                       include_smri_binary=True,
                                       include_age=True,
                                       include_sex=True):
    """
    ENHANCED compute_node_features with TRUE regional sMRI support from FreeSurfer.

    KEY IMPROVEMENT: Instead of broadcasting global sMRI values to all nodes,
    this function retrieves region-specific sMRI values from FreeSurfer data.
    """
    N = fc_tensor.shape[0]
    feature_list = []

    with torch.no_grad():
        # Connectivity features (unchanged)
        if include_connectivity:
            W = fc_tensor.abs().clone()
            W.fill_diagonal_(0.0)
            strength = W.mean(dim=1)
            deg = W.sum(dim=1)
            dmin, dmax = deg.min(), deg.max()
            degree_norm = (deg - dmin) / (dmax - dmin + 1e-6)
            W13 = torch.pow(W, 1.0/3.0)
            T = W13 @ W13 @ W13
            k = (W > 0).sum(dim=1).float()
            denom = k * (k - 1.0)
            clust = torch.zeros(N, dtype=torch.float32, device=W.device)
            m = denom > 0
            clust[m] = torch.diag(T)[m] / denom[m]
            feature_list.extend([strength, degree_norm, clust])

        # Demographics features (unchanged - still global)
        if include_demographics:
            if include_age:
                age_feat = torch.full((N,), float(age_z), dtype=torch.float32, device=fc_tensor.device)
                feature_list.append(age_feat)
            if include_sex:
                sex_feat = torch.full((N,), float(sex_bin), dtype=torch.float32, device=fc_tensor.device)
                feature_list.append(sex_feat)

        # APOE features (unchanged - still global)
        if include_apoe:
            e2_0_f = torch.full((N,), float(apoe2_0), dtype=torch.float32, device=fc_tensor.device)
            e2_1_f = torch.full((N,), float(apoe2_1), dtype=torch.float32, device=fc_tensor.device)
            e2_2_f = torch.full((N,), float(apoe2_2), dtype=torch.float32, device=fc_tensor.device)
            e4_0_f = torch.full((N,), float(apoe4_0), dtype=torch.float32, device=fc_tensor.device)
            e4_1_f = torch.full((N,), float(apoe4_1), dtype=torch.float32, device=fc_tensor.device)
            e4_2_f = torch.full((N,), float(apoe4_2), dtype=torch.float32, device=fc_tensor.device)
            feature_list.extend([e2_0_f, e2_1_f, e2_2_f, e4_0_f, e4_1_f, e4_2_f])

        # PET biomarkers (unchanged - still global)
        if include_pet:
            tau_feat = torch.full((N,), float(tau_positive), dtype=torch.float32, device=fc_tensor.device)
            amyloid_feat = torch.full((N,), float(amyloid_positive), dtype=torch.float32, device=fc_tensor.device)
            feature_list.extend([tau_feat, amyloid_feat])

        # ENHANCED sMRI features - TRUE REGIONAL SUPPORT
        if include_smri_continuous and REGIONAL_SMRI_AVAILABLE and participant_id is not None:
            try:
                # Get regional sMRI features for this participant
                regional_smri = freesurfer_extractor.get_regional_features(participant_id)
                if regional_smri is not None:
                    # Use TRUE regional sMRI values for each node
                    cortical_thickness = torch.tensor(regional_smri['cortical_thickness'],
                                                    dtype=torch.float32, device=fc_tensor.device)
                    surface_area = torch.tensor(regional_smri['surface_area'],
                                               dtype=torch.float32, device=fc_tensor.device)
                    volume = torch.tensor(regional_smri['volume'],
                                        dtype=torch.float32, device=fc_tensor.device)

                    feature_list.extend([cortical_thickness, surface_area, volume])
                    print(f"Using TRUE regional sMRI for participant {participant_id}")
                else:
                    # Fallback to global sMRI
                    print(f"No regional sMRI found for {participant_id}, using global values")
                    smri_hipp_f = torch.full((N,), float(smri_hippocampus_z), dtype=torch.float32, device=fc_tensor.device)
                    smri_ento_f = torch.full((N,), float(smri_entorhinal_z), dtype=torch.float32, device=fc_tensor.device)
                    smri_amyg_f = torch.full((N,), float(smri_amygdala_z), dtype=torch.float32, device=fc_tensor.device)
                    smri_gray_f = torch.full((N,), float(smri_gray_matter_z), dtype=torch.float32, device=fc_tensor.device)
                    smri_vent_f = torch.full((N,), float(smri_ventricles_z), dtype=torch.float32, device=fc_tensor.device)
                    smri_temp_f = torch.full((N,), float(smri_temporal_z), dtype=torch.float32, device=fc_tensor.device)
                    feature_list.extend([smri_hipp_f, smri_ento_f, smri_amyg_f, smri_gray_f, smri_vent_f, smri_temp_f])
            except Exception as e:
                print(f"Error getting regional sMRI for {participant_id}: {e}, using global values")
                smri_hipp_f = torch.full((N,), float(smri_hippocampus_z), dtype=torch.float32, device=fc_tensor.device)
                smri_ento_f = torch.full((N,), float(smri_entorhinal_z), dtype=torch.float32, device=fc_tensor.device)
                smri_amyg_f = torch.full((N,), float(smri_amygdala_z), dtype=torch.float32, device=fc_tensor.device)
                smri_gray_f = torch.full((N,), float(smri_gray_matter_z), dtype=torch.float32, device=fc_tensor.device)
                smri_vent_f = torch.full((N,), float(smri_ventricles_z), dtype=torch.float32, device=fc_tensor.device)
                smri_temp_f = torch.full((N,), float(smri_temporal_z), dtype=torch.float32, device=fc_tensor.device)
                feature_list.extend([smri_hipp_f, smri_ento_f, smri_amyg_f, smri_gray_f, smri_vent_f, smri_temp_f])
        elif include_smri_continuous:
            # Fallback to global sMRI (original approach)
            smri_hipp_f = torch.full((N,), float(smri_hippocampus_z), dtype=torch.float32, device=fc_tensor.device)
            smri_ento_f = torch.full((N,), float(smri_entorhinal_z), dtype=torch.float32, device=fc_tensor.device)
            smri_amyg_f = torch.full((N,), float(smri_amygdala_z), dtype=torch.float32, device=fc_tensor.device)
            smri_gray_f = torch.full((N,), float(smri_gray_matter_z), dtype=torch.float32, device=fc_tensor.device)
            smri_vent_f = torch.full((N,), float(smri_ventricles_z), dtype=torch.float32, device=fc_tensor.device)
            smri_temp_f = torch.full((N,), float(smri_temporal_z), dtype=torch.float32, device=fc_tensor.device)
            feature_list.extend([smri_hipp_f, smri_ento_f, smri_amyg_f, smri_gray_f, smri_vent_f, smri_temp_f])

        # sMRI binary (unchanged - still global)
        if include_smri_binary:
            smri_hatrophy_f = torch.full((N,), float(smri_hippocampal_atrophy), dtype=torch.float32, device=fc_tensor.device)
            smri_ethin_f = torch.full((N,), float(smri_entorhinal_thinning), dtype=torch.float32, device=fc_tensor.device)
            smri_venlarge_f = torch.full((N,), float(smri_ventricular_enlargement), dtype=torch.float32, device=fc_tensor.device)
            smri_adpat_f = torch.full((N,), float(smri_ad_pattern), dtype=torch.float32, device=fc_tensor.device)
            feature_list.extend([smri_hatrophy_f, smri_ethin_f, smri_venlarge_f, smri_adpat_f])

        if len(feature_list) == 0:
            raise ValueError("At least one feature group must be enabled!")

        x = torch.stack(feature_list, dim=1)

    return x


def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---- APOE genotype parser ----
def parse_apoe_dosage(g):
    """Parse APOE genotype string to dosage counts"""
    if not isinstance(g, str):
        return 0, 0, 0
    g = g.strip().replace('ε', '').replace('E', '')
    if '/' not in g:
        return 0, 0, 0
    a, b = g.split('/', 1)

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return None

    a, b = to_int(a), to_int(b)
    if a not in (2, 3, 4) or b not in (2, 3, 4):
        return 0, 0, 0

    e2 = (1 if a == 2 else 0) + (1 if b == 2 else 0)
    e3 = (1 if a == 3 else 0) + (1 if b == 3 else 0)
    e4 = (1 if a == 4 else 0) + (1 if b == 4 else 0)
    return e2, e3, e4


def assign_dk_network_ids():
    """Assign each of 112 DK nodes to a functional network (0-6)"""
    net_ids = torch.zeros(112, dtype=torch.long)

    # DMN (Default Mode Network)
    dmn_nodes = [21, 55, 23, 57, 12, 46, 6, 40, 13, 47, 7, 41]
    net_ids[dmn_nodes] = 0

    # Salience Network
    salience_nodes = [33, 67, 1, 35, 24, 58]
    net_ids[salience_nodes] = 1

    # Executive Control Network
    exec_nodes = [2, 36, 25, 59, 26, 60, 27, 61, 16, 50, 18, 52]
    net_ids[exec_nodes] = 2

    # Visual Network
    visual_nodes = [3, 37, 9, 43, 11, 45, 19, 53]
    net_ids[visual_nodes] = 3

    # Sensorimotor Network
    sensorimotor_nodes = [20, 54, 22, 56, 15, 49, 29, 63]
    net_ids[sensorimotor_nodes] = 4

    # Limbic Network
    limbic_nodes = [4, 38, 14, 48, 5, 39, 31, 65, 68, 69, 70, 71]
    net_ids[limbic_nodes] = 5

    # Subcortical
    subcortical_nodes = list(range(72, 112))
    net_ids[subcortical_nodes] = 6

    return net_ids

def select_feature_set_dk(x: torch.Tensor, feature_set: str) -> torch.Tensor:
    """
    DK Regional Feature Structure (19 dims):
      [0-2]   Connectivity: strength, degree_norm, clustering
      [3-4]   Demographics: age_z, sex_bin
      [5-10]  APOE: e2_0, e2_1, e2_2, e4_0, e4_1, e4_2
      [11-12] PET: tau_suvr, amyloid_suvr (regional)
      [13-14] sMRI continuous: volume_z, thickness_z (regional)
      [15-18] Binary: tau_pos, amy_pos, vol_atrophy, thick_atrophy
    """
    feature_set = (feature_set or "FULL").upper().strip()

    sets = {
        # === SINGLE MODALITY ===
        'CONNECTIVITY_ONLY': [0, 1, 2],
        'DEMO': [3, 4],
        'AGE_ONLY': [3],
        'SEX_ONLY': [4],
        'APOE_ONLY': [5, 6, 7, 8, 9, 10],
        'PET_ONLY': [11, 12, 15, 16],
        'TAU_ONLY': [11, 15],
        'AMYLOID_ONLY': [12, 16],
        'SMRI_CONTINUOUS_ONLY': [13, 14],
        'SMRI_BINARY_ONLY': [17, 18],
        'SMRI_ALL': [13, 14, 17, 18],

        # === PAIRWISE ===
        'SMRI_APOE': [5, 6, 7, 8, 9, 10, 13, 14, 17, 18],
        'SMRI_PET': [11, 12, 13, 14, 15, 16, 17, 18],
        'SMRI_CONNECTIVITY': [0, 1, 2, 13, 14, 17, 18],
        'APOE_PET': [5, 6, 7, 8, 9, 10, 11, 12, 15, 16],
        'CONNECTIVITY_PET': [0, 1, 2, 11, 12, 15, 16],
        'CONNECTIVITY_DEMO': [0, 1, 2, 3, 4],
        'CONNECTIVITY_APOE': [0, 1, 2, 5, 6, 7, 8, 9, 10],
        'DEMO_APOE': [3, 4, 5, 6, 7, 8, 9, 10],
        'DEMO_PET': [3, 4, 11, 12, 15, 16],
        'AGE_APOE': [3, 5, 6, 7, 8, 9, 10],
        'AGE_PET': [3, 11, 12, 15, 16],
        'AGE_SMRI': [3, 13, 14, 17, 18],
        'AGE_CONNECTIVITY': [0, 1, 2, 3],

        # === PROGRESSIVE (Triples/Quads) ===
        'DEMO_SMRI': [3, 4, 13, 14, 17, 18],
        'DEMO_SMRI_APOE': [3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18],
        'DEMO_PET_SMRI': [3, 4, 11, 12, 13, 14, 15, 16, 17, 18],
        'DEMO_APOE_PET': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16],
        'APOE_PET_SMRI': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        'CONNECTIVITY_SMRI_APOE': [0, 1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18],
        'CONNECTIVITY_SMRI_PET': [0, 1, 2, 11, 12, 13, 14, 15, 16, 17, 18],
        'CONNECTIVITY_DEMO_SMRI': [0, 1, 2, 3, 4, 13, 14, 17, 18],
        'DEMO_APOE_SMRI': [3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18],
        'AGE_SMRI_APOE': [3, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18],
        'AGE_SMRI_PET': [3, 11, 12, 13, 14, 15, 16, 17, 18],
        'AGE_CONNECTIVITY_SMRI': [0, 1, 2, 3, 13, 14, 17, 18],
        'DEMO_APOE_PET_SMRI': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],

        # === NEW MULTIMODAL COMBINATIONS ===
        'PET_AGE_SEX_APOE': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16],
        'SMRI_AGE_SEX_APOE': [3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18],
        'PET_SMRI_AGE_SEX_APOE': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],

        # === FULL ===
        'FULL': list(range(19)),
    }

    idx = sets.get(feature_set, sets['FULL'])
    return x[:, idx]

def extract_clinical_features_dk(meta, labels, feature_extractor, feature_set='DEMO_GENO_BIO'):
    """
    Extract subject-level features (no connectivity) for traditional ML models.

    Args:
        meta: Metadata dataframe
        labels: Label array
        feature_extractor: Regional feature extractor
        feature_set: Which features to include ('DEMO', 'DEMO_GENO', or 'DEMO_GENO_BIO')

    Returns: X [n_samples, n_features], y [n_samples]
    """
    n_samples = len(meta)

    # Extract demographics (always included)
    ages = meta['age'].values.astype(float)
    sexes = (meta['sex'].values == 'M').astype(float)
    ages_z = (ages - ages.mean()) / (ages.std() + 1e-6)

    features_list = [ages_z, sexes]  # Start with demographics

    # Add genetics if requested
    if 'GENO' in feature_set:
        if 'apoe_genotype' in meta.columns:
            geno_series = meta['apoe_genotype'].astype(str).fillna('')
            dosage_tuples = np.array([parse_apoe_dosage(g) for g in geno_series])
            e2_dos = dosage_tuples[:, 0].astype(int)
            e4_dos = dosage_tuples[:, 2].astype(int)
        elif 'apoe4_count' in meta.columns:
            e4_dos = meta['apoe4_count'].fillna(0).astype(float).round().clip(lower=0, upper=2).astype(int).values
            e2_dos = np.zeros(n_samples, dtype=int)
        else:
            e2_dos = np.zeros(n_samples, dtype=int)
            e4_dos = np.zeros(n_samples, dtype=int)

        # One-hot encode APOE
        apoe2_0 = (e2_dos == 0).astype(float)
        apoe2_1 = (e2_dos == 1).astype(float)
        apoe2_2 = (e2_dos == 2).astype(float)
        apoe4_0 = (e4_dos == 0).astype(float)
        apoe4_1 = (e4_dos == 1).astype(float)
        apoe4_2 = (e4_dos == 2).astype(float)

        features_list.extend([apoe2_0, apoe2_1, apoe2_2, apoe4_0, apoe4_1, apoe4_2])

    # Add biomarkers if requested
    if 'BIO' in feature_set:
        regional_features_list = []
        for i in range(n_samples):
            participant_id = meta.iloc[i]['participant_id']
            sess_id = meta.iloc[i]['session_id']
            rid = meta.iloc[i]['rid']

            regional_features = feature_extractor.extract_subject_regional_features(
                participant_id=participant_id,
                sess_id=sess_id,
                rid=rid
            )

            # Average biomarkers across all nodes
            # regional_features has keys: 'tau_suvr', 'amyloid_suvr', 'smri_volume_z', 'smri_thickness_z'
            tau_mean = regional_features['tau_suvr'].mean()
            amy_mean = regional_features['amyloid_suvr'].mean()
            vol_mean = regional_features['smri_volume_z'].mean()
            thick_mean = regional_features['smri_thickness_z'].mean()

            regional_features_list.append([tau_mean, amy_mean, vol_mean, thick_mean])

        regional_features_array = np.array(regional_features_list)

        # Add as separate columns
        for col_idx in range(regional_features_array.shape[1]):
            features_list.append(regional_features_array[:, col_idx])

    # Stack all features
    X = np.column_stack(features_list)

    return X, labels


def participant_stratified_split(meta_df, test_ratio=0.15, val_ratio=0.15, seed=42):
    """Split by participants, not sessions, to prevent data leakage"""
    from sklearn.model_selection import train_test_split

    print(f"\nPARTICIPANT-LEVEL SPLITTING")

    participant_labels = {}
    participant_session_counts = {}

    for _, row in meta_df.iterrows():
        pid = row['participant_id']
        diagnosis = row['diagnosis']

        if pid not in participant_labels:
            participant_labels[pid] = []
            participant_session_counts[pid] = 0

        participant_labels[pid].append(diagnosis)
        participant_session_counts[pid] += 1

    unique_participants = []
    participant_diagnoses = []

    for pid, diagnoses in participant_labels.items():
        unique_participants.append(pid)
        most_common = Counter(diagnoses).most_common(1)[0][0]
        participant_diagnoses.append(most_common)

    unique_participants = np.array(unique_participants)
    participant_diagnoses = np.array(participant_diagnoses)

    print(f"Total unique participants: {len(unique_participants)}")
    print(f"Participant diagnoses: CN={sum(participant_diagnoses=='CN')}, AD={sum(participant_diagnoses=='AD')}")

    label_map = {'CN': 0, 'AD': 1}
    participant_labels_numeric = np.array([label_map[d] for d in participant_diagnoses])

    train_val_participants, test_participants = train_test_split(
        unique_participants,
        test_size=test_ratio,
        stratify=participant_labels_numeric,
        random_state=seed
    )

    train_val_labels = np.array([label_map[participant_labels[p][0]] for p in train_val_participants])

    train_participants, val_participants = train_test_split(
        train_val_participants,
        test_size=val_ratio/(1-test_ratio),
        stratify=train_val_labels,
        random_state=seed
    )

    print(f"Participant splits:")
    print(f"  Train participants: {len(train_participants)}")
    print(f"  Val participants:   {len(val_participants)}")
    print(f"  Test participants:  {len(test_participants)}")

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, row in meta_df.iterrows():
        pid = row['participant_id']
        if pid in train_participants:
            train_indices.append(idx)
        elif pid in val_participants:
            val_indices.append(idx)
        elif pid in test_participants:
            test_indices.append(idx)

    print(f"Session splits:")
    print(f"  Train sessions: {len(train_indices)}")
    print(f"  Val sessions:   {len(val_indices)}")
    print(f"  Test sessions:  {len(test_indices)}")

    return np.array(train_indices), np.array(val_indices), np.array(test_indices)

def validate_graph_net_ids(graphs, require_net_ids=True):
    """Validate that all graphs have properly set net_ids"""
    if not graphs:
        return True

    # Check first graph to see if net_ids exist
    if not hasattr(graphs[0], 'net_ids'):
        if require_net_ids:
            raise ValueError(
                f"Graphs missing net_ids attribute. "
                f"This is required for network pooling mode."
            )
        else:
            print("Skipping net_ids validation (not required for global pooling)")
            return False

    # Validate all graphs
    for i, g in enumerate(graphs):
        if not hasattr(g, 'net_ids'):
            raise ValueError(f"Graph {i} missing net_ids attribute")
        if g.net_ids.shape[0] != g.num_nodes:
            raise ValueError(
                f"Graph {i}: net_ids size {g.net_ids.shape[0]} != num_nodes {g.num_nodes}"
            )
        if g.net_ids.min() < 0 or g.net_ids.max() > 6:
            raise ValueError(
                f"Graph {i}: invalid net_ids range [{g.net_ids.min()}, {g.net_ids.max()}]"
            )

    print(f"Validated net_ids for {len(graphs)} graphs")
    return True

def create_balanced_loader(graphs, batch_size, shuffle=True,
                          oversample_minority=True, pooling_mode='global'):
    """Create a DataLoader with balanced sampling"""
    from torch.utils.data import WeightedRandomSampler

    # Only validate net_ids if using network pooling
    if pooling_mode == 'network':
        validate_graph_net_ids(graphs, require_net_ids=True)
    else:
        validate_graph_net_ids(graphs, require_net_ids=False)

    if not oversample_minority or not shuffle:
        return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)

    labels = [g.y.item() for g in graphs]
    class_counts = Counter(labels)
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return DataLoader(graphs, batch_size=batch_size, sampler=sampler)

def train_traditional_ml(X_train, y_train, X_val, y_val, X_test, y_test, model_name='logistic'):
    """Train and evaluate traditional ML model on clinical features only"""
    from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

    if model_name == 'logistic':
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict probabilities
    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Optimize threshold on validation set
    best_threshold = 0.5
    best_bal_acc = 0.0
    for thresh in np.linspace(0.1, 0.9, 17):
        y_val_pred = (y_val_prob >= thresh).astype(int)
        tp = ((y_val_pred == 1) & (y_val == 1)).sum()
        tn = ((y_val_pred == 0) & (y_val == 0)).sum()
        fp = ((y_val_pred == 1) & (y_val == 0)).sum()
        fn = ((y_val_pred == 0) & (y_val == 1)).sum()
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        bal_acc = 0.5 * (sens + spec)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = thresh

    # Evaluate on test set
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    tp = ((y_test_pred == 1) & (y_test == 1)).sum()
    tn = ((y_test_pred == 0) & (y_test == 0)).sum()
    fp = ((y_test_pred == 1) & (y_test == 0)).sum()
    fn = ((y_test_pred == 0) & (y_test == 1)).sum()

    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    auc = roc_auc_score(y_test, y_test_prob) if len(np.unique(y_test)) > 1 else 0.5
    f1 = f1_score(y_test, y_test_pred) if y_test_pred.sum() > 0 else 0.0

    return {
        'auc': float(auc),
        'balanced_accuracy': float(balanced_accuracy),
        'f1': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'threshold': float(best_threshold)
    }


def participant_stratified_cv(meta_df, n_splits=5, seed=42):
    """5-fold CV splitting by participants to prevent data leakage"""
    from sklearn.model_selection import StratifiedKFold, train_test_split

    print(f"\nPARTICIPANT-LEVEL 5-FOLD CV")

    participant_labels = {}
    for _, row in meta_df.iterrows():
        pid = row['participant_id']
        diagnosis = row['diagnosis']

        if pid not in participant_labels:
            participant_labels[pid] = []
        participant_labels[pid].append(diagnosis)

    unique_participants = []
    participant_diagnoses = []

    for pid, diagnoses in participant_labels.items():
        unique_participants.append(pid)
        most_common = Counter(diagnoses).most_common(1)[0][0]
        participant_diagnoses.append(most_common)

    unique_participants = np.array(unique_participants)
    participant_diagnoses = np.array(participant_diagnoses)

    print(f"Total unique participants: {len(unique_participants)}")
    print(f"CN: {sum(participant_diagnoses=='CN')}, AD: {sum(participant_diagnoses=='AD')}")

    label_map = {'CN': 0, 'AD': 1}
    participant_labels_numeric = np.array([label_map[d] for d in participant_diagnoses])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cv_folds = []
    for fold_idx, (train_val_participants_idx, test_participants_idx) in enumerate(skf.split(unique_participants, participant_labels_numeric)):

        train_val_participants = unique_participants[train_val_participants_idx]
        test_participants = unique_participants[test_participants_idx]

        train_val_labels = participant_labels_numeric[train_val_participants_idx]

        train_idx_rel, val_idx_rel = train_test_split(
            range(len(train_val_participants)),
            test_size=0.2,
            stratify=train_val_labels,
            random_state=seed
        )

        train_participants = train_val_participants[train_idx_rel]
        val_participants = train_val_participants[val_idx_rel]

        train_indices = []
        val_indices = []
        test_indices = []

        for idx, row in meta_df.iterrows():
            pid = row['participant_id']
            if pid in train_participants:
                train_indices.append(idx)
            elif pid in val_participants:
                val_indices.append(idx)
            elif pid in test_participants:
                test_indices.append(idx)

        print(f"Fold {fold_idx+1}: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        cv_folds.append((
            np.array(train_indices),
            np.array(val_indices),
            np.array(test_indices)
        ))

    return cv_folds


def run_cv_experiment(experiment_name, config, device, feature_extractor, n_folds=5):

    """Run a 5-fold cross-validation experiment"""
    print(f"\n{'='*80}")
    print(f"5-FOLD CV EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")

    # Initialize trackers for this experiment
    learning_tracker = LearningCurveTracker()
    gradient_analyzer = GradientFlowAnalyzer()

    data_cfg = config.get('data')
    graph_cfg = config.get('graph')
    train_cfg = config.get('training')
    aug_cfg = config.get('augmentation')
    use_aug = bool(aug_cfg and aug_cfg.get('enabled', False))

    # Check if this is a non-graph experiment (clinical features only)
    is_clinical_only = config.get('clinical_only', False)
    save_participant_artifacts = config.get('save_participant_artifacts', False)

    # Create output directory for this experiment
    exp_output_dir = Path(config['output_dir']) / experiment_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dl = ADNIDataLoader(
        data_cfg['fc_matrices_dir'],
        data_cfg['demographics_file'],
        augmentation_config=aug_cfg if use_aug else None
    )
    _ = dl.load_demographics()
    fc_mats, meta = dl.load_fc_matrices()

    # CN vs AD only
    mask = meta['diagnosis'].isin(['CN','AD'])
    meta = meta[mask].reset_index(drop=True)
    fc_mats = fc_mats[mask.values]

    label_map = {'CN':0,'AD':1}
    labels = meta['diagnosis'].map(label_map).values.astype(int)

    # If this is a clinical-only experiment, extract features and use traditional ML
    if is_clinical_only:
        print(f"CLINICAL-ONLY EXPERIMENT (No FC connectivity)")
        feature_set = config.get('feature_set', 'DEMO_GENO_BIO')
        X_clinical, y_clinical = extract_clinical_features_dk(meta, labels, feature_extractor, feature_set)
        cv_folds = participant_stratified_cv(meta, n_splits=n_folds, seed=42)
        cv_results = {model_name: [] for model_name in config['models']}

        for fold, (train_idx, val_idx, test_idx) in enumerate(cv_folds):
            print(f"\n--- FOLD {fold + 1}/{n_folds} ---")

            X_train_clin = X_clinical[train_idx]
            X_val_clin = X_clinical[val_idx]
            X_test_clin = X_clinical[test_idx]
            y_train_clin = y_clinical[train_idx]
            y_val_clin = y_clinical[val_idx]
            y_test_clin = y_clinical[test_idx]

            for ml_model in config['models']:
                print(f"  Training {ml_model.upper()} (clinical only)")
                ml_results = train_traditional_ml(
                    X_train_clin, y_train_clin,
                    X_val_clin, y_val_clin,
                    X_test_clin, y_test_clin,
                    model_name=ml_model
                )

                fold_result = {
                    'fold': fold + 1,
                    'test_metrics': ml_results,
                    'model_type': 'clinical_only'
                }
                cv_results[ml_model].append(fold_result)

                print(f"    {ml_model.upper()}: AUC={ml_results['auc']:.4f}, "
                      f"BalAcc={ml_results['balanced_accuracy']:.4f}, "
                      f"F1={ml_results['f1']:.4f}",
                      f"SEN={test_metrics['sensitivity']:.4f}, ",
                      f"SPE={test_metrics['specificity']:.4f}")

        #After model training

        analyzer = EnhancedGNNInterpretability(model, device)

        # Overall importance
        grad_importance = analyzer.compute_gradient_attribution(test_loader)

        # Feature-specific
        if 'TAU' in config.get('feature_set', 'FULL'):
            tau_imp = analyzer.compute_feature_specific_importance(
                test_loader, [11], 'TAU'
            )

        # Disease pattern comparison
        comparison = analyzer.compare_to_disease_pattern(
            grad_importance['node_importance']
        )

        # Visualize
        output_dir = Path(config['output_dir'])
        analyzer.visualize_node_importance(
            grad_importance['node_importance'],
            output_dir / f'fold{fold+1}_importance.png',
            comparison_results=comparison
        )

        # Aggregate results
        aggregated_results = {}
        for model_name in config['models']:
            fold_results = cv_results[model_name]
            metrics = ['auc', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity']

            mean_metrics = {}
            std_metrics = {}

            for metric in metrics:
                values = [fold['test_metrics'][metric] for fold in fold_results]
                mean_metrics[metric] = np.mean(values)
                std_metrics[metric] = np.std(values)

            aggregated_results[model_name] = {
                'cv_mean': mean_metrics,
                'cv_std': std_metrics,
                'fold_results': fold_results,
                'n_folds': n_folds
            }

            print(f"\n{model_name.upper()} 5-FOLD CV RESULTS:")
            print(f"  AUC: {mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
            print(f"  Balanced Accuracy: {mean_metrics['balanced_accuracy']:.4f} ± {std_metrics['balanced_accuracy']:.4f}")
            print(f"  F1: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")

        return aggregated_results

    # Otherwise, it's a graph-based experiment
    # Build graphs with regional features
    gc = GraphConstructor(threshold=config['graph']['threshold'], top_k=config['graph']['top_k'])
    graphs = []

    # Check if we need network IDs (only for network pooling)
    needs_net_ids = config.get('pooling', {}).get('mode', 'global') == 'network'
    if needs_net_ids:
        print(f"Network pooling enabled - will add net_ids to all graphs")
    else:
        print(f"Global pooling enabled - net_ids not required")

    for i in range(len(fc_mats)):
        fc = torch.tensor(fc_mats[i], dtype=torch.float32)

        # Extract regional features
        participant_id = meta.iloc[i]['participant_id']
        sess_id = meta.iloc[i]['session_id']
        rid = meta.iloc[i]['rid']

        regional_features = feature_extractor.extract_subject_regional_features(
            participant_id=participant_id,
            sess_id=sess_id,
            rid=rid
        )

        # Get APOE dosages
        if 'apoe_genotype' in meta.columns:
            geno = meta.iloc[i]['apoe_genotype']
            e2_dos, _, e4_dos = parse_apoe_dosage(str(geno))
        else:
            e2_dos, e4_dos = 0, 0

        ages = meta['age'].values.astype(float)
        sexes = (meta['sex'].values == 'M').astype(float)
        ages_z = (ages - ages.mean()) / (ages.std() + 1e-6)

        apoe2_dos = (e2_dos == 0, e2_dos == 1, e2_dos == 2)
        apoe4_dos = (e4_dos == 0, e4_dos == 1, e4_dos == 2)

        if "broadcast" in experiment_name.lower():
            # Convert APOE dosages to one-hot for exp1 compatibility
            apoe2_0 = 1.0 if (e2_dos == 0) else 0.0
            apoe2_1 = 1.0 if (e2_dos == 1) else 0.0
            apoe2_2 = 1.0 if (e2_dos == 2) else 0.0
            apoe4_0 = 1.0 if (e4_dos == 0) else 0.0
            apoe4_1 = 1.0 if (e4_dos == 1) else 0.0
            apoe4_2 = 1.0 if (e4_dos == 2) else 0.0

            # Extract subject-level biomarker and sMRI features from metadata
            tau_pos = float(meta.iloc[i].get("tau_positive", 0.0))
            amy_pos = float(meta.iloc[i].get("amyloid_positive", 0.0))

            smri_hipp_z = float(meta.iloc[i].get("smri_hippocampus", 0.0))
            smri_ento_z = float(meta.iloc[i].get("smri_entorhinal", 0.0))
            smri_amyg_z = float(meta.iloc[i].get("smri_amygdala", 0.0))
            smri_gm_z = float(meta.iloc[i].get("smri_gray_matter", 0.0))
            smri_vent_z = float(meta.iloc[i].get("smri_ventricles", 0.0))
            smri_temp_z = float(meta.iloc[i].get("smri_temporal", 0.0))

            smri_hat = float(meta.iloc[i].get("smri_hippocampal_atrophy", 0.0))
            smri_ethin = float(meta.iloc[i].get("smri_entorhinal_thinning", 0.0))
            smri_venl = float(meta.iloc[i].get("smri_ventricular_enlargement", 0.0))
            smri_adpat = float(meta.iloc[i].get("smri_ad_pattern", 0.0))

            # Apply CN population z-scoring for continuous sMRI features
            cn_mask = (labels == 0)
            def z_score_cn(values, idx):
                cn_vals = values[cn_mask]
                cn_vals = cn_vals[cn_vals != 0]
                if len(cn_vals) > 0:
                    mu, sd = cn_vals.mean(), cn_vals.std()
                    if sd > 0:
                        return (values[idx] - mu) / sd
                return values[idx]

            # Get ages and sexes arrays for z-scoring
            ages = meta['age'].values.astype(float)
            sexes = (meta['sex'].values == 'M').astype(float)
            ages_z = (ages - ages.mean()) / (ages.std() + 1e-6)

            # Z-score sMRI using CN population
            smri_volumes = meta['smri_hippocampus'].fillna(0).astype(float).values
            smri_hipp_z = z_score_cn(smri_volumes, i) if smri_volumes[i] != 0 else 0.0

            smri_ento_volumes = meta['smri_entorhinal'].fillna(0).astype(float).values
            smri_ento_z = z_score_cn(smri_ento_volumes, i) if smri_ento_volumes[i] != 0 else 0.0

            smri_amyg_volumes = meta['smri_amygdala'].fillna(0).astype(float).values
            smri_amyg_z = z_score_cn(smri_amyg_volumes, i) if smri_amyg_volumes[i] != 0 else 0.0

            smri_gm_volumes = meta['smri_gray_matter'].fillna(0).astype(float).values
            smri_gm_z = z_score_cn(smri_gm_volumes, i) if smri_gm_volumes[i] != 0 else 0.0

            smri_vent_volumes = meta['smri_ventricles'].fillna(0).astype(float).values
            smri_vent_z = z_score_cn(smri_vent_volumes, i) if smri_vent_volumes[i] != 0 else 0.0

            smri_temp_volumes = meta['smri_temporal'].fillna(0).astype(float).values
            smri_temp_z = z_score_cn(smri_temp_volumes, i) if smri_temp_volumes[i] != 0 else 0.0

            # Get feature configuration for ablation studies
            feature_config = config.get("feature_groups", {
                "include_connectivity": True,
                "include_demographics": True,
                "include_apoe": True,
                "include_pet": True,
                "include_smri_continuous": True,
                "include_smri_binary": True,
                "include_age": True,
                "include_sex": True,
            })

            # Regional sMRI for regional_smri_* experiments, broadcast global sMRI otherwise
            if "regional_smri" in experiment_name.lower():
                # Use enhanced function with TRUE regional sMRI
                participant_id = meta.iloc[i].get('participant_id', meta.iloc[i].get('Participant_ID', f'unknown_{i}'))
                x = compute_node_features_enhanced_smri(
                    fc_tensor=fc,
                    age_z=ages_z[i],
                    sex_bin=sexes[i],
                    apoe2_0=apoe2_0, apoe2_1=apoe2_1, apoe2_2=apoe2_2,
                    apoe4_0=apoe4_0, apoe4_1=apoe4_1, apoe4_2=apoe4_2,
                    tau_positive=tau_pos, amyloid_positive=amy_pos,
                    smri_hippocampus_z=smri_hipp_z, smri_entorhinal_z=smri_ento_z, smri_amygdala_z=smri_amyg_z,
                    smri_gray_matter_z=smri_gm_z, smri_ventricles_z=smri_vent_z, smri_temporal_z=smri_temp_z,
                    smri_hippocampal_atrophy=smri_hat, smri_entorhinal_thinning=smri_ethin,
                    smri_ventricular_enlargement=smri_venl, smri_ad_pattern=smri_adpat,
                    participant_id=participant_id,
                    **feature_config
                )
            else:
                # Use broadcast feature engineering from exp1 (original)
                x = compute_node_features(
                    fc_tensor=fc,
                    age_z=ages_z[i],
                    sex_bin=sexes[i],
                    apoe2_0=apoe2_0, apoe2_1=apoe2_1, apoe2_2=apoe2_2,
                    apoe4_0=apoe4_0, apoe4_1=apoe4_1, apoe4_2=apoe4_2,
                    tau_positive=tau_pos, amyloid_positive=amy_pos,
                    smri_hippocampus_z=smri_hipp_z, smri_entorhinal_z=smri_ento_z, smri_amygdala_z=smri_amyg_z,
                    smri_gray_matter_z=smri_gm_z, smri_ventricles_z=smri_vent_z, smri_temporal_z=smri_temp_z,
                    smri_hippocampal_atrophy=smri_hat, smri_entorhinal_thinning=smri_ethin,
                    smri_ventricular_enlargement=smri_venl, smri_ad_pattern=smri_adpat,
                    **feature_config
                )
        else:
            # Keep original regional feature engineering from exp2
            x = compute_node_features_dk_regional(
                fc_tensor=fc,
                regional_features=regional_features,
                age_z=ages_z[i],
                sex_bin=sexes[i],
                apoe2_dos=apoe2_dos,
                apoe4_dos=apoe4_dos
            )

        # Apply feature selection for ablation studies
        x = select_feature_set_dk(x, config.get('feature_set', 'FULL'))

        g = gc.construct_graph(fc.numpy(), x)
        g.y = torch.tensor(labels[i], dtype=torch.long)
        g.num_nodes = fc.shape[0]

        # Add network IDs ONLY if using network pooling
        if needs_net_ids:
            dk_net_ids = assign_dk_network_ids()
            num_nodes = fc.shape[0]

            # Truncate or pad network IDs to match number of nodes
            if num_nodes <= 112:
                g.net_ids = dk_net_ids[:num_nodes].clone()
            else:
                # If more than 112 nodes (shouldn't happen with DK), extend with subcortical
                extra_ids = torch.full((num_nodes - 112,), 6, dtype=torch.long)
                g.net_ids = torch.cat([dk_net_ids, extra_ids])

            # Ensure net_ids match the number of nodes
            assert g.net_ids.shape[0] == num_nodes, \
                f"net_ids size {g.net_ids.shape[0]} doesn't match num_nodes {num_nodes}"

        graphs.append(g)

    # 5-fold Cross Validation
    cv_folds = participant_stratified_cv(meta, n_splits=n_folds, seed=42)
# === Split Persistence for Reproducibility Between Variants ===
    split_file = Path(config['output_dir']).parent / "shared_cv_splits.json"
    split_file.parent.mkdir(parents=True, exist_ok=True)

    def _folds_to_serializable(cv_folds, meta_df):
        """Convert fold indices to participant IDs for cross-variant reproducibility"""
        serializable_folds = []
        for (train_idx, val_idx, test_idx) in cv_folds:
            fold_data = {
                "train_participants": meta_df.iloc[train_idx]['participant_id'].tolist(),
                "val_participants": meta_df.iloc[val_idx]['participant_id'].tolist(),
                "test_participants": meta_df.iloc[test_idx]['participant_id'].tolist(),
            }
            serializable_folds.append(fold_data)
        return serializable_folds

    def _serializable_to_folds(serializable_folds, meta_df):
        """Reconstruct fold indices from participant IDs"""
        pid_to_idx = {pid: idx for idx, pid in enumerate(meta_df['participant_id'])}
        reconstructed_folds = []

        for fold_data in serializable_folds:
            train_idx = np.array([pid_to_idx[pid] for pid in fold_data["train_participants"] if pid in pid_to_idx])
            val_idx = np.array([pid_to_idx[pid] for pid in fold_data["val_participants"] if pid in pid_to_idx])
            test_idx = np.array([pid_to_idx[pid] for pid in fold_data["test_participants"] if pid in pid_to_idx])
            reconstructed_folds.append((train_idx, val_idx, test_idx))

        return reconstructed_folds

    if split_file.exists():
        try:
            with open(split_file, "r") as f:
                saved_splits = json.load(f)
            cv_folds = _serializable_to_folds(saved_splits["folds"], meta)
            print(f"Reusing existing CV splits from {split_file}")
            print(f"  Original experiment: {saved_splits.get('source_experiment', 'unknown')}")
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Corrupted CV splits file detected ({e})")
            print(f"   Deleting corrupted file and creating new splits...")
            split_file.unlink()  # Delete corrupted file

            # Create new splits
            serializable_folds = _folds_to_serializable(cv_folds, meta)
            split_data = {
                "source_experiment": experiment_name,
                "n_folds": n_folds,
                "seed": 42,
                "folds": serializable_folds
            }
            with open(split_file, "w") as f:
                json.dump(split_data, f, indent=2)
            print(f"Created new CV splits and saved to {split_file}")
    else:
        serializable_folds = _folds_to_serializable(cv_folds, meta)
        split_data = {
            "source_experiment": experiment_name,
            "n_folds": n_folds,
            "seed": 42,
            "folds": serializable_folds
        }
        with open(split_file, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved CV splits to {split_file} for cross-variant reproducibility")
        serializable_folds = _folds_to_serializable(cv_folds, meta)
        split_data = {
            "source_experiment": experiment_name,
            "n_folds": n_folds,
            "seed": 42,
            "folds": serializable_folds
        }
        with open(split_file, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved CV splits to {split_file} for cross-variant reproducibility")
    # === End Split Persistence ===
    cv_results = {model_name: [] for model_name in config['models']}

    # Storage for interpretability across folds
    all_fold_importance = {model_name: [] for model_name in config['models']}

    for fold, (train_idx, val_idx, test_idx) in enumerate(cv_folds):
        print(f"\n--- FOLD {fold + 1}/{n_folds} ---")

        fold_dir = exp_output_dir / f'fold_{fold+1}'
        fold_dir.mkdir(exist_ok=True)

        train_graphs = [graphs[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        if use_aug:
            train_graphs, train_labels = dl.augment_minority_class(train_graphs, train_labels)

        val_graphs = [graphs[i] for i in val_idx]
        test_graphs = [graphs[i] for i in test_idx]

        print(f"Fold {fold+1}: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
        print(f"Train labels: {Counter(train_labels)}")

        pooling_mode = config.get('pooling', {}).get('mode', 'global')

        train_loader = create_balanced_loader(
            train_graphs,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            oversample_minority=config['training']['balanced_sampling'],
            pooling_mode=pooling_mode
        )
        val_loader = DataLoader(val_graphs, batch_size=config['training']['batch_size'], shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=config['training']['batch_size'], shuffle=False)

        for model_name in config['models']:
            print(f"  Training {model_name.upper()} (Fold {fold+1})")

            # Create model directory
            model_fold_dir = fold_dir / model_name
            model_fold_dir.mkdir(exist_ok=True)

            input_dim = train_graphs[0].num_node_features

            if model_name == 'gkan':
                model = SimpleGKAN(
                    input_dim=input_dim, hidden_dim=32, num_classes=2,
                    grid_size=8, spline_order=3, dropout=0.5, residual=True,
                    pooling_mode=config.get('pooling', {}).get('mode', 'global')
                ).to(device)
                train_fun = train_model_optimized
            elif model_name == 'gcn':
                model = GCNBaseline(
                    input_dim=input_dim, hidden_dims=(160, 160, 80),
                    num_classes=2, dropout=0.5,
                    pooling_mode=config.get('pooling', {}).get('mode', 'global')
                ).to(device)
                train_fun = train_model
            elif model_name == 'gat':
                model = GATBaseline(
                    input_dim=input_dim, num_classes=2,
                    hidden_channels=64, heads=4, num_layers=3, dropout=0.5,
                    pooling_mode=config.get('pooling', {}).get('mode', 'global')
                ).to(device)
                train_fun = train_model
            else:
                continue

            train_info = train_model_with_tracking(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config['training'],
                device=device,
                learning_curve_tracker=learning_tracker,
                gradient_analyzer=gradient_analyzer,
                save_dir=model_fold_dir / 'training'  #  CHANGED THIS LINE
            )

            # Temperature & threshold tuning (keep existing)
            T = fit_temperature(model, val_loader, device)

            tune_metric = "youden" if model_name == 'gcn' else "balanced_acc"
            tuned_t, _ = tune_threshold_on_val(
                model, val_loader, device,
                temperature=T, metric=tune_metric,
                grid=np.linspace(0.05, 0.95, 19),
                min_sensitivity=None,
                min_specificity=None,
                prevalence_tolerance=0.25
            )

            if not np.isfinite(tuned_t) or tuned_t <= 0.0:
                tuned_t = 0.05
            if tuned_t >= 0.95:
                tuned_t = 0.90

            test_metrics = evaluate_model_with_interpretability(
                model=model,
                model_name=model_name,
                test_loader=test_loader,
                device=device,
                threshold=tuned_t,
                temperature=T,
                run_interpretability=True,  #  ALWAYS TRUE NOW
                output_dir=model_fold_dir  #  CHANGED THIS LINE
            )

            # Extract node importance for aggregation
            if (model_fold_dir / 'interpretability' / 'importance_results.pkl').exists():
                import pickle
                with open(model_fold_dir / 'interpretability' / 'importance_results.pkl', 'rb') as f:
                    imp_results = pickle.load(f)
                    if 'integrated_gradients' in imp_results:
                        node_imp = imp_results['integrated_gradients']['node_importance_mean']
                        all_fold_importance[model_name].append(node_imp)

            # Extract GAT attention weights (participant-level, opt-in only)
            if model_name == 'gat' and save_participant_artifacts:
                interp = EnhancedGNNInterpretability(model, device)
                attention_maps = interp.extract_attention_weights(test_loader)

                if attention_maps:
                    import pickle
                    save_path = model_fold_dir / 'attention_weights.pkl'
                    with open(save_path, 'wb') as f:
                        pickle.dump(attention_maps, f)
                    print(f"    Saved GAT attention weights to {save_path}")


            # Comprehensive results
            fold_result = {
                'fold': fold + 1,
                'experiment': experiment_name,
                'model': model_name,
                'training': {
                    'best_val_auc': train_info['best_val_auc'],
                    'best_val_loss': train_info['best_val_loss'],
                    'final_epoch': train_info['final_epoch']
                },
                'calibration': {
                    'temperature': float(T),
                    'threshold': float(tuned_t)
                },
                'test_metrics': test_metrics,
                'model_dir': str(model_fold_dir),
                'interpretability_dir': test_metrics.get('interpretability_dir', None)
            }
            cv_results[model_name].append(fold_result)

            # Save fold-specific results

            with open(model_fold_dir / 'fold_results.json', 'w') as f:
                json.dump(fold_result, f, indent=2)

            print(f"    Fold {fold+1} {model_name.upper()}: AUC={test_metrics['auc']:.4f}, "
                  f"BalAcc={test_metrics['balanced_accuracy']:.4f}, "
                  f"F1={test_metrics['f1']:.4f}",
                  f"SEN={test_metrics['sensitivity']:.4f}, "
                  f"SPE={test_metrics['specificity']:.4f}")

    # Aggregate results
    aggregated_results = {}
    for model_name in config['models']:
        if model_name not in cv_results:
            continue

        fold_results = cv_results[model_name]
        metrics = ['auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity']

        mean_metrics = {}
        std_metrics = {}

        for metric in metrics:
            values = [fold['test_metrics'][metric] for fold in fold_results]
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)

        aggregated_results[model_name] = {
            'cv_mean': mean_metrics,
            'cv_std': std_metrics,
            'fold_results': fold_results,
            'n_folds': n_folds
        }

        print(f"\n{model_name.upper()} 5-FOLD CV RESULTS:")
        print(f"  AUC: {mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
        print(f"  Balanced Accuracy: {mean_metrics['balanced_accuracy']:.4f} ± {std_metrics['balanced_accuracy']:.4f}")
        print(f"  F1: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")

    # Aggregate importance across folds
    print(f"\n{'='*60}")
    print(f"AGGREGATING INTERPRETABILITY RESULTS")
    print(f"{'='*60}")

    aggregated_importance = {}
    for model_name in config['models']:
        if all_fold_importance[model_name]:
            importance_array = np.array(all_fold_importance[model_name])

            aggregated_importance[model_name] = {
                'mean_importance': importance_array.mean(axis=0),
                'std_importance': importance_array.std(axis=0),
                'per_fold': importance_array
            }

            # Save aggregated importance
            np.save(
                exp_output_dir / f'{model_name}_aggregated_importance.npy',
                aggregated_importance[model_name]['mean_importance']
            )
    # Create cross-fold visualizations
    if aggregated_importance:
        create_cross_fold_analysis(
            cv_results=cv_results,
            aggregated_importance=aggregated_importance,
            output_dir=exp_output_dir,
            experiment_name=experiment_name,
            device=device
        )

    # Cross-model comparison
    if len(config['models']) > 1:
        comparison_dir = exp_output_dir / 'model_comparison'
        comparison_dir.mkdir(exist_ok=True)

        create_model_comparison_report(
            cv_results=cv_results,
            aggregated_results=aggregated_results,
            output_dir=comparison_dir,
            experiment_name=experiment_name
        )

    return aggregated_results

    # Create cross-fold visualizations
    if aggregated_importance:
        create_cross_fold_analysis(
            cv_results=cv_results,
            aggregated_importance=aggregated_importance,
            output_dir=exp_output_dir,
            experiment_name=experiment_name,
            device=device
        )

    return aggregated_results

    # === AFTER ALL FOLDS: Create comparison visualizations ===
    if len(config['models']) > 1:
        comparison_dir = Path(config['output_dir']) / 'cross_fold_comparison'
        comparison_dir.mkdir(exist_ok=True)

        # Plot learning curves across folds
        learning_tracker.plot_metric_comparison(
            metrics=['val_auc', 'val_balanced_accuracy', 'val_f1'],
            output_path=comparison_dir / 'metrics_evolution.png',
            title=f'{experiment_name} - Cross-Model Comparison'
        )

def create_feature_importance_ranking(all_results, output_dir):
    """
    Comprehensive analysis including:
    1. Regional vs Broadcast feature engineering comparison
    2. Synergy analysis for feature combinations
    3. Brain region importance ranking from interpretability
    """
    import numpy as np
    from pathlib import Path

    def get_auc(exp_name, model_key):
        try:
            return float(all_results[exp_name][model_key]['cv_mean']['auc'])
        except Exception:
            return np.nan

    # Experiment mappings
    # Regional experiments (original)
    # singles_regional = {
    #     'conn': ('ablation_connectivity_only', 'Connectivity'),
    #     'demo': ('ablation_demographics_only', 'Demographics'),
    #     'age': ('ablation_age_only', 'Age'),
    #     'sex': ('ablation_sex_only', 'Sex'),
    #     'apoe': ('ablation_apoe_only', 'APOE'),
    #     'pet': ('ablation_pet_only', 'PET'),
    #     'smri_c': ('ablation_smri_continuous_only', 'sMRI Continuous'),
    #     'smri_b': ('ablation_smri_binary_only', 'sMRI Binary'),
    #     'smri': ('ablation_smri_all', 'sMRI (All)'),
    # }

    # Broadcast experiments
    #singles_broadcast = {
    #     'conn': ('ablation_broadcast_connectivity_only', 'Connectivity'),
    #     'demo': ('ablation_broadcast_demographics_only', 'Demographics'),
    #     'age': ('ablation_broadcast_age_only', 'Age'),
    #     'sex': ('ablation_broadcast_sex_only', 'Sex'),
    #     'apoe': ('ablation_broadcast_apoe_only', 'APOE'),
    #     'pet': ('ablation_broadcast_pet_only', 'PET'),
    #     'smri_c': ('ablation_broadcast_smri_continuous_only', 'sMRI Continuous'),
    #     'smri_b': ('ablation_broadcast_smri_binary_only', 'sMRI Binary'),
    #     'smri': ('ablation_broadcast_smri_all', 'sMRI (All)'),
    # }

    ##### new

    singles_regional = {
        'conn': ('ablation_connectivity_only', 'Connectivity'),
        'demo': ('ablation_demographics_only', 'Demographics'),
        'apoe': ('ablation_apoe_only', 'APOE'),
        'pet': ('ablation_pet_only', 'PET'),
        'smri_c': ('ablation_smri_continuous_only', 'sMRI Continuous'),
        'smri_b': ('ablation_smri_binary_only', 'sMRI Binary'),
        'smri': ('ablation_smri_all', 'sMRI (All)'),
    }

    singles_broadcast = {
        'conn': ('ablation_broadcast_connectivity_only', 'Connectivity'),
        'demo': ('ablation_broadcast_demographics_only', 'Demographics'),
        'apoe': ('ablation_broadcast_apoe_only', 'APOE'),
        'pet': ('ablation_broadcast_pet_only', 'PET'),
        'smri_c': ('ablation_broadcast_smri_continuous_only', 'sMRI Continuous'),
        'smri_b': ('ablation_broadcast_smri_binary_only', 'sMRI Binary'),
        'smri': ('ablation_broadcast_smri_all', 'sMRI (All)'),
    }


    # Pairwise combinations
    pairs_regional = {
        'smri+apoe': 'ablation_smri_apoe',
        'smri+pet': 'ablation_smri_pet',
        'smri+conn': 'ablation_smri_connectivity',
        'apoe+pet': 'ablation_apoe_pet',
        'conn+pet': 'ablation_connectivity_pet',
        'conn+demo': 'ablation_connectivity_demo',
        'conn+apoe': 'ablation_connectivity_apoe',
        'demo+apoe': 'ablation_demo_apoe',
        'demo+pet': 'ablation_demo_pet',
        'age+apoe': 'ablation_age_apoe',
        'age+pet': 'ablation_age_pet',
        'age+smri': 'ablation_age_smri',
        'age+conn': 'ablation_age_connectivity',
    }

    pairs_broadcast = {
        'smri+apoe': 'ablation_broadcast_smri_apoe',
        'smri+pet': 'ablation_broadcast_smri_pet',
        'smri+conn': 'ablation_broadcast_smri_connectivity',
        'apoe+pet': 'ablation_broadcast_apoe_pet',
        'conn+pet': 'ablation_broadcast_connectivity_pet',
        'conn+demo': 'ablation_broadcast_connectivity_demo',
        'conn+apoe': 'ablation_broadcast_connectivity_apoe',
        'demo+apoe': 'ablation_broadcast_demo_apoe',
        'demo+pet': 'ablation_broadcast_demo_pet',
        'age+apoe': 'ablation_broadcast_age_apoe',
        'age+pet': 'ablation_broadcast_age_pet',
        'age+smri': 'ablation_broadcast_age_smri',
        'age+conn': 'ablation_broadcast_age_connectivity',
    }

    # Progressive combinations
    progressive_regional = {
        'demo+smri': 'ablation_demo_smri',
        'demo+smri+apoe': 'ablation_demo_smri_apoe',
        'demo+pet+smri': 'ablation_demo_pet_smri',
        'demo+apoe+pet': 'ablation_demo_apoe_pet',
        'apoe+pet+smri': 'ablation_apoe_pet_smri',
        'conn+smri+apoe': 'ablation_connectivity_smri_apoe',
        'conn+smri+pet': 'ablation_connectivity_smri_pet',
        'conn+demo+smri': 'ablation_connectivity_demo_smri',
        'demo+apoe+smri': 'ablation_demo_apoe_smri',
        'age+smri+apoe': 'ablation_age_smri_apoe',
        'age+smri+pet': 'ablation_age_smri_pet',
        'age+conn+smri': 'ablation_age_connectivity_smri',
        'demo+apoe+pet+smri': 'ablation_demo_apoe_pet_smri',
        'full': 'ablation_full_multimodal',
    }

    progressive_broadcast = {
        'demo+smri': 'ablation_broadcast_demo_smri',
        'demo+smri+apoe': 'ablation_broadcast_demo_smri_apoe',
        'demo+pet+smri': 'ablation_broadcast_demo_pet_smri',
        'demo+apoe+pet': 'ablation_broadcast_demo_apoe_pet',
        'apoe+pet+smri': 'ablation_broadcast_apoe_pet_smri',
        'conn+smri+apoe': 'ablation_broadcast_connectivity_smri_apoe',
        'conn+smri+pet': 'ablation_broadcast_connectivity_smri_pet',
        'conn+demo+smri': 'ablation_broadcast_connectivity_demo_smri',
        'demo+apoe+smri': 'ablation_broadcast_demo_apoe_smri',
        'age+smri+apoe': 'ablation_broadcast_age_smri_apoe',
        'age+smri+pet': 'ablation_broadcast_age_smri_pet',
        'age+conn+smri': 'ablation_broadcast_age_connectivity_smri',
        'demo+apoe+pet+smri': 'ablation_broadcast_demo_apoe_pet_smri',
        'full': 'ablation_broadcast_full_multimodal',
    }

    #model_sets = [('gkan', 'GKAN'), ('gcn', 'GCN'), ('gat', 'GAT')]
    model_sets = [('gat', 'GAT')]

    # Create comprehensive analysis
    ranking_path = Path(output_dir) / "comprehensive_feature_analysis.txt"
    with open(ranking_path, "w") as f:
        f.write("FEATURE ENGINEERING AND INTERPRETABILITY ANALYSIS\n")
        f.write("="*80 + "\n\n")

        # Section 1: regional vs broadcast comparison
        f.write("SECTION 1: REGIONAL vs BROADCAST FEATURE ENGINEERING\n")
        f.write("="*60 + "\n\n")

        for model_key, model_name in model_sets:
            f.write(f"[{model_name} - Single Modality Comparison]\n")
            f.write(f"{'Feature':<20} {'Regional':>10} {'Broadcast':>10} {'Δ (B-R)':>12} {'Winner':>10}\n")
            f.write("-"*80 + "\n")

            regional_wins = 0
            broadcast_wins = 0

            for tag, (exp_reg, nice) in singles_regional.items():
                exp_broad = singles_broadcast[tag][0]
                auc_reg = get_auc(exp_reg, model_key)
                auc_broad = get_auc(exp_broad, model_key)
                diff = auc_broad - auc_reg if not (np.isnan(auc_reg) or np.isnan(auc_broad)) else np.nan

                auc_reg_str = "NaN" if np.isnan(auc_reg) else f"{auc_reg:.4f}"
                auc_broad_str = "NaN" if np.isnan(auc_broad) else f"{auc_broad:.4f}"
                diff_str = "NaN" if np.isnan(diff) else f"{diff:+.4f}"

                if not np.isnan(diff):
                    if diff > 0.005:  # Broadcast wins by >0.5%
                        winner = "Broadcast"
                        broadcast_wins += 1
                    elif diff < -0.005:  # Regional wins by >0.5%
                        winner = "Regional"
                        regional_wins += 1
                    else:
                        winner = "Tie"
                else:
                    winner = "N/A"

                f.write(f"{nice:<20} {auc_reg_str:>10} {auc_broad_str:>10} {diff_str:>12} {winner:>10}\n")

            f.write(f"\nSUMMARY: Regional={regional_wins}, Broadcast={broadcast_wins}\n\n")

        # Section 2: single modality rankings
        f.write("SECTION 2: SINGLE MODALITY RANKINGS\n")
        f.write("="*60 + "\n\n")

        for approach, singles_dict in [("Regional", singles_regional), ("Broadcast", singles_broadcast)]:
            f.write(f"[{approach} Approach Rankings]\n")
            for model_key, model_name in model_sets:
                f.write(f"\n{model_name}:\n")
                scores = []
                for tag, (exp, nice) in singles_dict.items():
                    scores.append((nice, get_auc(exp, model_key)))
                scores_sorted = sorted(scores, key=lambda x: (-1 if np.isnan(x[1]) else -x[1], x[0]))

                f.write(f"{'Rank':<5} {'Feature':<20} {'AUC':>8}\n")
                f.write("-"*40 + "\n")
                for rank, (feat, val) in enumerate(scores_sorted, 1):
                    auc_str = "NaN" if np.isnan(val) else f"{val:.4f}"
                    f.write(f"{rank:<5} {feat:<20} {auc_str:>8}\n")
            f.write("\n")

        # Section 3: synergy analysis
        f.write("SECTION 3: SYNERGY ANALYSIS\n")
        f.write("="*60 + "\n")
        f.write("Synergy = AUC(combination) - max(AUC(individual_components))\n\n")

        def calculate_synergy(pairs_dict, singles_dict, approach_name):
            f.write(f"[{approach_name} - Pairwise Synergies]\n")
            for model_key, model_name in model_sets:
                f.write(f"\n{model_name}:\n")
                f.write(f"{'Pair':<20} {'Combined':>10} {'Best_Single':>12} {'Synergy':>10}\n")
                f.write("-"*60 + "\n")

                def get_single_auc(tag):
                    mapping = {
                        'smri': singles_dict['smri'][0], 'apoe': singles_dict['apoe'][0],
                        'pet': singles_dict['pet'][0], 'conn': singles_dict['conn'][0],
                        'demo': singles_dict['demo'][0], 'age': singles_dict['age'][0]
                    }
                    return get_auc(mapping.get(tag, singles_dict['demo'][0]), model_key)

                for name, exp in pairs_dict.items():
                    auc_pair = get_auc(exp, model_key)
                    components = name.split('+')
                    single_aucs = [get_single_auc(comp) for comp in components]
                    best_single = np.nanmax(single_aucs) if single_aucs else np.nan
                    synergy = auc_pair - best_single if not (np.isnan(auc_pair) or np.isnan(best_single)) else np.nan

                    pair_str = f"{auc_pair:.4f}" if not np.isnan(auc_pair) else "NaN"
                    single_str = f"{best_single:.4f}" if not np.isnan(best_single) else "NaN"
                    synergy_str = f"{synergy:+.4f}" if not np.isnan(synergy) else "NaN"

                    f.write(f"{name:<20} {pair_str:>10} {single_str:>12} {synergy_str:>10}\n")
            f.write("\n")

        calculate_synergy(pairs_regional, singles_regional, "Regional")
        calculate_synergy(pairs_broadcast, singles_broadcast, "Broadcast")

        # Section 4: progressive combinations
        f.write("SECTION 4: PROGRESSIVE COMBINATION ANALYSIS\n")
        f.write("="*60 + "\n\n")

        for approach, prog_dict in [("Regional", progressive_regional), ("Broadcast", progressive_broadcast)]:
            f.write(f"[{approach} - Progressive Combinations]\n")
            for model_key, model_name in model_sets:
                f.write(f"\n{model_name}:\n")
                f.write(f"{'Combination':<25} {'AUC':>8}\n")
                f.write("-"*40 + "\n")

                prog_scores = []
                for name, exp in prog_dict.items():
                    auc = get_auc(exp, model_key)
                    prog_scores.append((name, auc))

                prog_scores_sorted = sorted(prog_scores, key=lambda x: (-1 if np.isnan(x[1]) else -x[1], x[0]))

                for name, auc in prog_scores_sorted:
                    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "NaN"
                    f.write(f"{name:<25} {auc_str:>8}\n")
            f.write("\n")

        # Section 5: best overall comparison
        f.write("SECTION 5: BEST OVERALL PERFORMANCE COMPARISON\n")
        f.write("="*60 + "\n\n")

        f.write("Full Multimodal Performance:\n")
        f.write(f"{'Model':<10} {'Regional':>10} {'Broadcast':>10} {'Δ (B-R)':>12} {'Best':>10}\n")
        f.write("-"*60 + "\n")

        for model_key, model_name in model_sets:
            auc_reg = get_auc('ablation_full_multimodal', model_key)
            auc_broad = get_auc('ablation_broadcast_full_multimodal', model_key)
            diff = auc_broad - auc_reg if not (np.isnan(auc_reg) or np.isnan(auc_broad)) else np.nan

            auc_reg_str = f"{auc_reg:.4f}" if not np.isnan(auc_reg) else "NaN"
            auc_broad_str = f"{auc_broad:.4f}" if not np.isnan(auc_broad) else "NaN"
            diff_str = f"{diff:+.4f}" if not np.isnan(diff) else "NaN"

            if not np.isnan(diff):
                best = "Broadcast" if diff > 0 else "Regional" if diff < 0 else "Tie"
            else:
                best = "N/A"

            f.write(f"{model_name:<10} {auc_reg_str:>10} {auc_broad_str:>10} {diff_str:>12} {best:>10}\n")

        # Section 6: roi interpretability analysis
        f.write("\n\nSECTION 6: BRAIN REGION IMPORTANCE ANALYSIS\n")
        f.write("="*60 + "\n\n")

        # Check if interpretability results exist
        importance_files = []
        base_output_dir = Path(output_dir)

        # Look for aggregated importance files
        for exp_name in ['ablation_full_multimodal', 'ablation_broadcast_full_multimodal']:
            exp_dir = base_output_dir / exp_name
            if exp_dir.exists():
                for model_name in ['gkan', 'gcn', 'gat']:
                    importance_file = exp_dir / f'{model_name}_aggregated_importance.npy'
                    if importance_file.exists():
                        importance_files.append((exp_name, model_name, importance_file))

        if importance_files:
            f.write("Brain Region Importance Rankings (Top 20 regions):\n\n")

            # Define region names (112 DK regions)
            region_names = [
                "Left_Bankssts", "Left_Caudal_Anterior_Cingulate", "Left_Caudal_Middle_Frontal",
                "Left_Cuneus", "Left_Entorhinal", "Left_Fusiform", "Left_Inferior_Parietal",
                "Left_Inferior_Temporal", "Left_Insula", "Left_Isthmus_Cingulate",
                "Left_Lateral_Occipital", "Left_Lateral_Orbitofrontal", "Left_Lingual",
                "Left_Medial_Orbitofrontal", "Left_Middle_Temporal", "Left_Parahippocampal",
                "Left_Paracentral", "Left_Pars_Opercularis", "Left_Pars_Orbitalis",
                "Left_Pars_Triangularis", "Left_Pericalcarine", "Left_Postcentral",
                "Left_Posterior_Cingulate", "Left_Precentral", "Left_Precuneus",
                "Left_Rostral_Anterior_Cingulate", "Left_Rostral_Middle_Frontal",
                "Left_Superior_Frontal", "Left_Superior_Parietal", "Left_Superior_Temporal",
                "Left_Supramarginal", "Left_Frontal_Pole", "Left_Temporal_Pole",
                "Left_Transverse_Temporal", "Right_Bankssts", "Right_Caudal_Anterior_Cingulate",
                "Right_Caudal_Middle_Frontal", "Right_Cuneus", "Right_Entorhinal",
                "Right_Fusiform", "Right_Inferior_Parietal", "Right_Inferior_Temporal",
                "Right_Insula", "Right_Isthmus_Cingulate", "Right_Lateral_Occipital",
                "Right_Lateral_Orbitofrontal", "Right_Lingual", "Right_Medial_Orbitofrontal",
                "Right_Middle_Temporal", "Right_Parahippocampal", "Right_Paracentral",
                "Right_Pars_Opercularis", "Right_Pars_Orbitalis", "Right_Pars_Triangularis",
                "Right_Pericalcarine", "Right_Postcentral", "Right_Posterior_Cingulate",
                "Right_Precentral", "Right_Precuneus", "Right_Rostral_Anterior_Cingulate",
                "Right_Rostral_Middle_Frontal", "Right_Superior_Frontal", "Right_Superior_Parietal",
                "Right_Superior_Temporal", "Right_Supramarginal", "Right_Frontal_Pole",
                "Right_Temporal_Pole", "Right_Transverse_Temporal", "Left_Accumbens",
                "Left_Amygdala", "Left_Caudate", "Left_Hippocampus", "Left_Pallidum",
                "Left_Putamen", "Left_Thalamus", "Right_Accumbens", "Right_Amygdala",
                "Right_Caudate", "Right_Hippocampus", "Right_Pallidum", "Right_Putamen",
                "Right_Thalamus"
            ]

            # Ensure we have enough region names
            while len(region_names) < 112:
                region_names.append(f"Region_{len(region_names)}")

            for exp_name, model_name, importance_file in importance_files:
                approach = "Regional" if "broadcast" not in exp_name else "Broadcast"
                f.write(f"[{approach} - {model_name.upper()}]\n")

                try:
                    importance = np.load(importance_file)

                    # Get top 20 regions
                    top_k = 20
                    if len(importance) >= top_k:
                        top_indices = np.argsort(importance)[-top_k:][::-1]

                        f.write(f"{'Rank':<5} {'Region':<30} {'Importance':>12}\n")
                        f.write("-"*50 + "\n")

                        for rank, idx in enumerate(top_indices, 1):
                            region_name = region_names[idx] if idx < len(region_names) else f"Region_{idx}"
                            f.write(f"{rank:<5} {region_name:<30} {importance[idx]:>12.6f}\n")
                    else:
                        f.write("Insufficient importance data available.\n")

                except Exception as e:
                    f.write(f"Error loading importance data: {str(e)}\n")

                f.write("\n")
        else:
            f.write("No interpretability results found. Run experiments with interpretability enabled.\n")

        # Section 7: summary recommendations
        f.write("\nSECTION 7: SUMMARY & RECOMMENDATIONS\n")
        f.write("="*60 + "\n\n")

        f.write("KEY FINDINGS:\n")
        f.write("1. Feature Engineering Approach: [Compare Regional vs Broadcast]\n")
        f.write("2. Most Important Individual Features: [Based on single modality rankings]\n")
        f.write("3. Best Feature Combinations: [Based on synergy analysis]\n")
        f.write("4. Model Performance: [Best performing model-approach combination]\n")
        f.write("5. Brain Regions: [Most important regions for AD classification]\n\n")

        f.write("RECOMMENDATIONS:\n")
        f.write("- Use the approach (Regional/Broadcast) that consistently performs better\n")
        f.write("- Focus on feature combinations showing positive synergy\n")
        f.write("- Consider interpretability when choosing between similar-performing approaches\n")
        f.write("- Validate findings on independent test sets\n")

    logger.info(f"Comprehensive feature analysis saved to: {ranking_path}")

    # Create summary csv for easy analysis
    import pandas as pd

    summary_data = []
    for approach, singles_dict in [("Regional", singles_regional), ("Broadcast", singles_broadcast)]:
        for model_key, model_name in model_sets:
            for tag, (exp, nice) in singles_dict.items():
                auc = get_auc(exp, model_key)
                summary_data.append({
                    'Approach': approach,
                    'Model': model_name,
                    'Feature': nice,
                    'Experiment': exp,
                    'AUC': auc
                })

    # Add full multimodal results
    for model_key, model_name in model_sets:
        for approach, exp in [("Regional", "ablation_full_multimodal"), ("Broadcast", "ablation_broadcast_full_multimodal")]:
            auc = get_auc(exp, model_key)
            summary_data.append({
                'Approach': approach,
                'Model': model_name,
                'Feature': 'Full_Multimodal',
                'Experiment': exp,
                'AUC': auc
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(Path(output_dir) / "feature_engineering_summary.csv", index=False)

    logger.info(f"Summary CSV saved to: {Path(output_dir) / 'feature_engineering_summary.csv'}")

def get_input_dim(include_connectivity=True, include_demographics=True, include_apoe=True,
                  include_pet=True, include_smri_continuous=True, include_smri_binary=True,
                  include_age=True, include_sex=True):
    """Calculate input dimension based on enabled feature groups"""
    dim = 0
    if include_connectivity:
        dim += 3
    if include_demographics:
        if include_age:
            dim += 1
        if include_sex:
            dim += 1
    if include_apoe:
        dim += 6
    if include_pet:
        dim += 2
    if include_smri_continuous:
        dim += 6
    if include_smri_binary:
        dim += 4
    return dim

def get_feature_config_name(include_connectivity, include_demographics, include_apoe,
                            include_pet, include_smri_continuous, include_smri_binary,
                            include_age=True, include_sex=True):
    """Generate descriptive name for feature configuration"""
    parts = []
    if include_connectivity:
        parts.append("conn")
    if include_demographics:
        if include_age and include_sex:
            parts.append("demo")
        elif include_age:
            parts.append("demo_age")
        elif include_sex:
            parts.append("demo_sex")
    if include_apoe:
        parts.append("apoe")
    if include_pet:
        parts.append("pet")
    if include_smri_continuous:
        parts.append("smri_cont")
    if include_smri_binary:
        parts.append("smri_bin")
    return "_".join(parts) if parts else "no_features"


def build_experiment_configs(config):
    """Expand the YAML config into one resolved config dict per experiment.

    The experiment name drives feature-engineering branches inside
    run_cv_experiment (substrings "broadcast" and "regional_smri"), so the
    names from the config are preserved verbatim as dictionary keys.
    """
    base = {
        'data': config['data'],
        'graph': config.get('graph', {'threshold': None, 'top_k': None}),
        'training': config['training'],
        'models': config['models'],
        'output_dir': config['output_dir'],
        'save_participant_artifacts': config.get('save_participant_artifacts', False),
    }

    experiments = {}
    for exp in config['experiments']:
        name = exp['name']

        # Per-experiment training overrides on top of the shared training block.
        exp_training = dict(base['training'])
        if 'balanced_sampling' in exp:
            exp_training['balanced_sampling'] = exp['balanced_sampling']
        if 'focal_gamma' in exp:
            exp_training['focal_gamma'] = exp['focal_gamma']
        exp_training.update(exp.get('training', {}))

        experiments[name] = {
            **base,
            'feature_set': exp['feature_set'],
            'pooling': exp.get('pooling', {'mode': 'global'}),
            'augmentation': exp.get('augmentation', {'enabled': False}),
            'output_dir': str(Path(base['output_dir']) / name),
            'training': exp_training,
        }
    return experiments


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seeds(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    data_cfg = config['data']

    # Optionally initialize the FreeSurfer regional sMRI extractor. This requires
    # both the module (shipped) and a FreeSurfer regional CSV (provided by the
    # user). Without them, the runner uses global/proxy sMRI features.
    global freesurfer_extractor
    if REGIONAL_SMRI_AVAILABLE and data_cfg.get('freesurfer_file'):
        freesurfer_extractor = FreeSurferRegionalExtractor(
            freesurfer_file=data_cfg['freesurfer_file'],
            demographics_file=data_cfg.get('freesurfer_demographics_file'),
        )
        logger.info("FreeSurfer regional sMRI extractor initialized.")
    else:
        freesurfer_extractor = None
        logger.info("Using global/proxy sMRI features (regional FreeSurfer extractor not configured).")

    # Regional feature extractor backed by pre-extracted regional PET data.
    feature_extractor = DKRegionalFeatureExtractor(
        regional_pet_file=data_cfg['regional_pet_file'],
        regional_smri_file=data_cfg.get('regional_smri_file'),
        demographics_file=data_cfg.get('demographics_file'),
    )

    experiments = build_experiment_configs(config)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_folds = config.get('n_folds', 5)

    all_results = {}
    for exp_name, exp_config in experiments.items():
        all_results[exp_name] = run_cv_experiment(
            exp_name, exp_config, device, feature_extractor, n_folds=n_folds
        )

    # Aggregate ROI importance ranking and cross-experiment comparison table.
    create_feature_importance_ranking(all_results, output_dir)
    create_cv_comparison_csv(all_results, output_dir)

    with open(output_dir / "all_experiments_5cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("All cross-validation experiments completed.")
    logger.info(f"Results written to: {output_dir}")


def create_cv_comparison_csv(all_results, output_dir):
    """Create comprehensive comparison CSV for cross-validation results"""
    import csv

    csv_path = output_dir / "experiments_5cv_comparison.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Experiment", "Model", "Type", "AUC_Mean", "AUC_Std",
            "Balanced_Accuracy_Mean", "Balanced_Accuracy_Std",
            "F1_Mean", "F1_Std", "Sensitivity_Mean", "Sensitivity_Std",
            "Specificity_Mean", "Specificity_Std", "N_Folds"
        ])

        for exp_name, exp_results in all_results.items():
            exp_type = "Clinical-Only" if "CLINICAL" in exp_name else "Graph-Based"

            for model_name, results in exp_results.items():
                m = results['cv_mean']
                s = results['cv_std']

                writer.writerow([
                    exp_name, model_name, exp_type,
                    f"{m['auc']:.4f}", f"{s['auc']:.4f}",
                    f"{m['balanced_accuracy']:.4f}", f"{s['balanced_accuracy']:.4f}",
                    f"{m['f1']:.4f}", f"{s['f1']:.4f}",
                    f"{m['sensitivity']:.4f}", f"{s['sensitivity']:.4f}",
                    f"{m['specificity']:.4f}", f"{s['specificity']:.4f}",
                    results['n_folds']
                ])

    print(f"5-fold CV comparison table saved to: {csv_path}")

def create_model_comparison_report(cv_results, aggregated_results, output_dir, experiment_name):
    """
    Create comparative analysis across models for CV experiments.
    """
    import matplotlib.pyplot as plt
    import pandas as pd


    output_dir = Path(output_dir)
    print(f"\nCreating cross-model comparison for {experiment_name}...")

    models = list(aggregated_results.keys())
    metrics_to_compare = ['auc', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity']

    # 1. create comparison table
    comparison_data = []
    for model_name in models:
        cv_mean = aggregated_results[model_name]['cv_mean']
        cv_std = aggregated_results[model_name]['cv_std']

        for metric in metrics_to_compare:
            comparison_data.append({
                'Model': model_name.upper(),
                'Metric': metric.replace('_', ' ').title(),
                'Mean': cv_mean.get(metric, 0),
                'Std': cv_std.get(metric, 0)
            })

    df = pd.DataFrame(comparison_data)

    # 2. create comparison plot
    fig, ax = plt.subplots(figsize=(14, 7))

    pivot_mean = df.pivot(index='Metric', columns='Model', values='Mean')
    pivot_std = df.pivot(index='Metric', columns='Model', values='Std')

    # Bar plot with error bars
    pivot_mean.plot(kind='bar', ax=ax, width=0.8, edgecolor='black',
                    yerr=pivot_std, capsize=4, alpha=0.8)

    ax.set_ylabel('Score (Mean ± SD)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Metric', fontweight='bold', fontsize=12)
    ax.set_title(f'{experiment_name} - Cross-Model Performance (5-Fold CV)',
                fontweight='bold', fontsize=14)
    ax.legend(title='Model', fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_cv.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. save comparison table
    comparison_table = df.pivot_table(
        index='Metric',
        columns='Model',
        values=['Mean', 'Std']
    )
    comparison_table.to_csv(output_dir / 'model_comparison_cv.csv')

    # 4. create summary
    summary = {
        'experiment': experiment_name,
        'best_model_by_metric': {},
        'model_rankings': {}
    }

    # Best model per metric
    for metric in metrics_to_compare:
        metric_values = {
            model: aggregated_results[model]['cv_mean'].get(metric, 0)
            for model in models
        }
        best_model = max(metric_values.items(), key=lambda x: x[1])
        summary['best_model_by_metric'][metric] = {
            'model': best_model[0],
            'mean': float(best_model[1]),
            'std': float(aggregated_results[best_model[0]]['cv_std'].get(metric, 0))
        }

    # Overall ranking (by average AUC + Balanced Accuracy)
    model_scores = {}
    for model in models:
        auc = aggregated_results[model]['cv_mean'].get('auc', 0)
        bal_acc = aggregated_results[model]['cv_mean'].get('balanced_accuracy', 0)
        model_scores[model] = 0.5 * (auc + bal_acc)

    ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    summary['model_rankings'] = [
        {
            'rank': i+1,
            'model': model,
            'avg_score': float(score),
            'auc_mean': float(aggregated_results[model]['cv_mean'].get('auc', 0)),
            'bal_acc_mean': float(aggregated_results[model]['cv_mean'].get('balanced_accuracy', 0))
        }
        for i, (model, score) in enumerate(ranked_models)
    ]

    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Cross-model comparison saved to: {output_dir}")
    print(f"\nBEST OVERALL MODEL: {ranked_models[0][0].upper()} (score: {ranked_models[0][1]:.4f})")

def create_cross_fold_analysis(cv_results, aggregated_importance, output_dir, experiment_name, device):
    """Create comprehensive cross-fold analysis with importance ranking."""
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir = Path(output_dir)
    print(f"\nCreating cross-fold analysis for {experiment_name}...")

    # Get region names using a lightweight analyzer instance
    from gkan_simple_v25 import SimpleGKAN

    dummy_model = SimpleGKAN(input_dim=19, hidden_dim=32, num_classes=2)
    analyzer = EnhancedGNNInterpretability(dummy_model, device)
    region_names = analyzer.region_names

    for model_name, importance_data in aggregated_importance.items():
        mean_importance = importance_data['mean_importance']
        std_importance = importance_data['std_importance']

        # Get top 30 regions
        top_k = 30
        top_indices = np.argsort(mean_importance)[-top_k:][::-1]

        # Create DataFrame
        top_regions_df = pd.DataFrame({
            'Rank': range(1, top_k + 1),
            'Region': [region_names[i] for i in top_indices],
            'Mean_Importance': mean_importance[top_indices],
            'Std_Importance': std_importance[top_indices],
            'CV': std_importance[top_indices] / (mean_importance[top_indices] + 1e-10)
        })

        # Save CSV
        top_regions_df.to_csv(
            output_dir / f'{model_name}_top_regions_cv.csv',
            index=False
        )

        # Visualize with error bars
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(top_indices))
        regions = [region_names[i][:30] for i in top_indices]
        values = mean_importance[top_indices]
        errors = std_importance[top_indices]

        # Color by AD pathology
        ad_keywords = ['Entorhinal', 'Hippocampus', 'Parahippocampal',
                      'Precuneus', 'Cingulate', 'Temporal', 'Amygdala']
        colors = ['#d62728' if any(kw in r for kw in ad_keywords) else '#1f77b4'
                 for r in regions]

        ax.barh(y_pos, values, xerr=errors, color=colors, alpha=0.7,
               edgecolor='black', capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(regions, fontsize=9)
        ax.set_xlabel('Importance (Mean ± SD)', fontweight='bold', fontsize=11)
        ax.set_title(f'{model_name.upper()}: Top {top_k} Regions (5-Fold CV)',
                    fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_top_regions_cv.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved {model_name} cross-fold analysis")


def analyze_fold_stability(fold_results, output_dir, model_name):
    """Analyze metric stability across folds."""
    import pandas as pd

    metrics = ['auc', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity']

    stability_data = []
    for metric in metrics:
        values = [fold['test_metrics'][metric] for fold in fold_results]
        stability_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Mean': np.mean(values),
            'Std': np.std(values),
            'CV': np.std(values) / (np.mean(values) + 1e-10),
            'Min': np.min(values),
            'Max': np.max(values)
        })

    stability_df = pd.DataFrame(stability_data)
    stability_df.to_csv(output_dir / f'{model_name}_fold_stability.csv', index=False)


if __name__ == "__main__":
    main()
