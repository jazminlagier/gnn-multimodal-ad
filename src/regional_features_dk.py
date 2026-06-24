"""
Desikan-Killiany Regional Features Extractor
==============================================================
- Uses pre-extracted regional PET data (participants_sessions_regional_pet_cn_ad.csv)
- Proper per-region TAU and Amyloid SUVR values for each of 112 DK nodes
- Returns dense feature matrices
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path
# DK_34, DK2ST_L, DK2ST_R map the 34 Desikan-Killiany cortical labels to the
# ADNI UCSF FreeSurfer (UCSFFSX) ST column codes per hemisphere. They are defined
# in an extended dk_region_mapping module. When that mapping is not available,
# the regional sMRI path falls back to proxy/global sMRI features (see below).
try:
    from dk_region_mapping import DK_34, DK2ST_L, DK2ST_R
    ST_MAPPING_AVAILABLE = True
except ImportError:
    DK_34, DK2ST_L, DK2ST_R = [], {}, {}
    ST_MAPPING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Modes:
#   "broadcast_pet_regional_smri" (default) -> PET comes from global columns (no regional PET),
#                                             sMRI stays regional (UCSFFSX7 STxx*).
#   "regional_pet_regional_smri" -> PET uses regional LH/RH + bilateral fallback,
#                                   sMRI stays regional.
import os
REGIONAL_MODE = os.environ.get("REGIONAL_MODE", "broadcast_pet_regional_smri").strip()
VALID_MODES = {"broadcast_pet_regional_smri", "regional_pet_regional_smri"}
if REGIONAL_MODE not in VALID_MODES:
    logger.warning(f"Unknown REGIONAL_MODE={REGIONAL_MODE}, defaulting to 'broadcast_pet_regional_smri'")
    REGIONAL_MODE = "broadcast_pet_regional_smri"
logger.info(f"REGIONAL_MODE = {REGIONAL_MODE}")


# Import the DK region mapping
try:
    from dk_region_mapping import REGION_TO_IDX, IDX_TO_REGION, TOTAL_REGIONS
except ImportError:
    logger.warning("dk_region_mapping.py not found, using internal mapping")
    TOTAL_REGIONS = 112  # Standard DK parcellation


class DKRegionalFeatureExtractor:
    """
    Extract per-region features for Desikan-Killiany parcellation (112 nodes)

    Uses pre-extracted regional PET data to provide:
    - Regional TAU SUVR values (per node)
    - Regional Amyloid SUVR values (per node)
    - Regional sMRI features (volume, thickness)

    This dramatically increases PET coverage compared to using only global values.
    """

    def __init__(self,
            regional_pet_file: str,
            regional_smri_file: Optional[str] = None,
            demographics_file: Optional[str] = None,
            mode: str = "hybrid"):

        """
        Initialize feature extractor with pre-extracted regional PET data.

        Args:
            regional_pet_file: Path to participants_sessions_regional_pet_cn_ad.csv
            demographics_file: Optional path to demographics (for compatibility)
        """
        logger.info("="*80)
        logger.info("INITIALIZING REGIONAL FEATURE EXTRACTOR")
        logger.info("="*80)

        # Load pre-extracted regional PET data
        self.regional_pet_df = pd.read_csv(regional_pet_file)
        # (NEW) Load regional sMRI if provided
        self.regional_smri_df = None
        if regional_smri_file and Path(regional_smri_file).exists():
            # read with low_memory=False to avoid dtype warnings / mixed types
            self.regional_smri_df = pd.read_csv(regional_smri_file, low_memory=False)
            df = self.regional_smri_df

            # --- normalize IDs so we can match PET schema (participant_id, session_id) ---
            # participant_id: from PTID like "AAA_S_BBBB" -> "sub-ADNIAAASBBBB"
            if 'participant_id' not in df.columns:
                if 'PTID' in df.columns:
                    pid_norm = (
                        df['PTID'].astype(str)
                        .str.replace('_S_', 'S', regex=False)  # "AAA_S_BBBB" -> "AAASBBBB"
                        .str.replace('_', '', regex=False)     # remove any remaining underscores
                    )
                    df['participant_id'] = 'sub-ADNI' + pid_norm
                else:
                    # keep column present (avoids KeyError later)
                    df['participant_id'] = None

            # session_id: from VISCODE2 like "m72" -> "ses-M72", "scmri" -> "ses-SCMRI"
            if 'session_id' not in df.columns:
                if 'VISCODE2' in df.columns:
                    df['session_id'] = 'ses-' + df['VISCODE2'].astype(str).str.upper()
                else:
                    df['session_id'] = None

            self.regional_smri_df = df
            logger.info(f"Loaded regional sMRI data: {len(self.regional_smri_df)} sessions")
        else:
            logger.info("No regional sMRI file provided; will fall back to proxy/global sMRI if needed")

        logger.info(f"Loaded regional PET data: {len(self.regional_pet_df)} sessions")

        # Load demographics if provided (for compatibility)
        if demographics_file and Path(demographics_file).exists():
            self.demographics_df = pd.read_csv(demographics_file)
            logger.info(f"Loaded demographics: {len(self.demographics_df)} records")
        else:
            self.demographics_df = None
            logger.info("No demographics file loaded")

        # Load region mapping
        self.region_mapping = self._load_region_mapping()
        logger.info(f"Loaded region mapping: {len(self.region_mapping)} regions")

        # Compute CN normalization statistics
        self.cn_stats = self._compute_cn_statistics()
        logger.info(f"Computed CN normalization stats: {len(self.cn_stats)} features")

        # Log data availability
        self._log_data_availability()

        logger.info("="*80)

    def _load_region_mapping(self) -> Dict[str, int]:
        """
        Load mapping from region names to DK node indices (0-111).

        Returns:
            Dictionary mapping region names to node indices
        """
        try:
            from dk_region_mapping import REGION_TO_IDX
            return REGION_TO_IDX
        except ImportError:
            # Fallback: create basic mapping for 112 DK nodes
            logger.warning("Using fallback region mapping")
            mapping = {}

            # Left hemisphere cortical (0-33)
            lh_regions = [
                "CTX_LH_BANKSSTS", "CTX_LH_CAUDALANTERIORCINGULATE",
                "CTX_LH_CAUDALMIDDLEFRONTAL", "CTX_LH_CUNEUS",
                "CTX_LH_ENTORHINAL", "CTX_LH_FRONTALPOLE", "CTX_LH_FUSIFORM",
                "CTX_LH_INFERIORPARIETAL", "CTX_LH_INFERIORTEMPORAL",
                "CTX_LH_INSULA", "CTX_LH_ISTHMUSCINGULATE",
                "CTX_LH_LATERALOCCIPITAL", "CTX_LH_LATERALORBITOFRONTAL",
                "CTX_LH_LINGUAL", "CTX_LH_MEDIALORBITOFRONTAL",
                "CTX_LH_MIDDLETEMPORAL", "CTX_LH_PARACENTRAL",
                "CTX_LH_PARAHIPPOCAMPAL", "CTX_LH_PARSOPERCULARIS",
                "CTX_LH_PARSORBITALIS", "CTX_LH_PARSTRIANGULARIS",
                "CTX_LH_PERICALCARINE", "CTX_LH_POSTCENTRAL",
                "CTX_LH_POSTERIORCINGULATE", "CTX_LH_PRECENTRAL",
                "CTX_LH_PRECUNEUS", "CTX_LH_ROSTRALANTERIORCINGULATE",
                "CTX_LH_ROSTRALMIDDLEFRONTAL", "CTX_LH_SUPERIORFRONTAL",
                "CTX_LH_SUPERIORPARIETAL", "CTX_LH_SUPERIORTEMPORAL",
                "CTX_LH_SUPRAMARGINAL", "CTX_LH_TEMPORALPOLE",
                "CTX_LH_TRANSVERSETEMPORAL"
            ]
            for i, region in enumerate(lh_regions[:34]):
                mapping[region] = i

            # Right hemisphere cortical (34-67)
            rh_regions = [r.replace("LH", "RH") for r in lh_regions]
            for i, region in enumerate(rh_regions[:34]):
                mapping[region] = i + 34

            # Subcortical (68-111)
            subcortical = [
                "LEFT_HIPPOCAMPUS", "RIGHT_HIPPOCAMPUS",
                "LEFT_AMYGDALA", "RIGHT_AMYGDALA",
                "LEFT_CAUDATE", "RIGHT_CAUDATE",
                "LEFT_PUTAMEN", "RIGHT_PUTAMEN",
                "LEFT_PALLIDUM", "RIGHT_PALLIDUM",
                "LEFT_THALAMUS_PROPER", "RIGHT_THALAMUS_PROPER",
            ]
            for i, region in enumerate(subcortical):
                if 68 + i < 112:
                    mapping[region] = 68 + i

            # Fill remaining slots
            for i in range(len(mapping), 112):
                mapping[f"NODE_{i}"] = i

            return mapping

    def get_freesurfer_regional_data(self, participant_id: str, sess_id: Optional[str] = None):
        """
        Return a dict for the UCSFFSX7 row matching this participant/session,
        containing STxx{CV,TA,SA} columns.
        """
        try:
            if self.regional_smri_df is None:
                return None
            df = self.regional_smri_df

            mask = df['participant_id'].eq(participant_id)
            if ('session_id' in df.columns) and df['session_id'].notna().any() and (sess_id is not None):
                mask &= df['session_id'].eq(sess_id)

            if not mask.any():
                return None

            return df.loc[mask].iloc[0].to_dict()
        except Exception as e:
            print(f"Error loading FreeSurfer data for {participant_id} {sess_id}: {e}")
            return None


    def _compute_cn_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute CN population statistics for z-score normalization.

        Returns:
            Dictionary of mean and std for each regional feature
        """
        cn_mask = self.regional_pet_df['diagnosis'] == 'CN'
        cn_data = self.regional_pet_df[cn_mask]

        logger.info(f"Computing CN stats from {len(cn_data)} CN subjects")

        stats = {}

        # Compute stats for each region's features
        for region_name, node_idx in self.region_mapping.items():
            # TAU stats
            tau_col = f'TAU_{region_name}_SUVR'
            if tau_col in cn_data.columns:
                values = cn_data[tau_col].dropna()
                if len(values) > 10:  # Need sufficient samples
                    stats[f'tau_{node_idx}'] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()) if values.std() > 0 else 1.0
                    }

            # Amyloid stats
            amy_col = f'AMYLOID_{region_name}_SUVR'
            if amy_col in cn_data.columns:
                values = cn_data[amy_col].dropna()
                if len(values) > 10:
                    stats[f'amyloid_{node_idx}'] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()) if values.std() > 0 else 1.0
                    }

        # sMRI stats (global features)
        smri_features = [
            'SMRI_HIPPOCAMPUS_BILATERAL',
            'SMRI_ENTORHINAL_BILATERAL',
            'SMRI_AMYGDALA_BILATERAL',
            'SMRI_FUSIFORM_THICKNESS_LEFT',  # Use as proxy
            'SMRI_VENTRICLES',
            'SMRI_TOTAL_GRAY_MATTER'
        ]

        for feat in smri_features:
            if feat in cn_data.columns:
                values = cn_data[feat].dropna()
                if len(values) > 10:
                    stats[feat] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()) if values.std() > 0 else 1.0
                    }

        logger.info(f"Computed stats for {len(stats)} regional features")
        return stats

    def _log_data_availability(self):
        """Log data availability statistics"""
        logger.info("\n" + "="*80)
        logger.info("DATA AVAILABILITY SUMMARY")
        logger.info("="*80)

        total = len(self.regional_pet_df)

        # Check tau availability
        if 'tau_data_available' in self.regional_pet_df.columns:
            tau_avail = self.regional_pet_df['tau_data_available'].sum()
        else:
            # Count rows with any TAU data
            tau_cols = [col for col in self.regional_pet_df.columns if col.startswith('TAU_')]
            tau_avail = self.regional_pet_df[tau_cols].notna().any(axis=1).sum()

        # Check amyloid availability
        if 'amyloid_data_available' in self.regional_pet_df.columns:
            amy_avail = self.regional_pet_df['amyloid_data_available'].sum()
        else:
            amy_cols = [col for col in self.regional_pet_df.columns if col.startswith('AMYLOID_')]
            amy_avail = self.regional_pet_df[amy_cols].notna().any(axis=1).sum()

        # Check sMRI availability
        if 'smri_data_available' in self.regional_pet_df.columns:
            smri_avail = self.regional_pet_df['smri_data_available'].sum()
        else:
            smri_cols = [col for col in self.regional_pet_df.columns if col.startswith('SMRI_')]
            smri_avail = self.regional_pet_df[smri_cols].notna().any(axis=1).sum()

        # Both PET modalities
        both_pet = 0
        if 'both_pet_available' in self.regional_pet_df.columns:
            both_pet = self.regional_pet_df['both_pet_available'].sum()

        logger.info(f"Total sessions: {total}")
        logger.info(f"  TAU PET:     {tau_avail:4d} ({100*tau_avail/total:5.1f}%)")
        logger.info(f"  Amyloid PET: {amy_avail:4d} ({100*amy_avail/total:5.1f}%)")
        logger.info(f"  sMRI:        {smri_avail:4d} ({100*smri_avail/total:5.1f}%)")
        logger.info(f"  Both PET:    {both_pet:4d} ({100*both_pet/total:5.1f}%)")

        # Diagnosis breakdown
        if 'diagnosis' in self.regional_pet_df.columns:
            logger.info(f"\nDiagnosis distribution:")
            for dx in self.regional_pet_df['diagnosis'].unique():
                count = (self.regional_pet_df['diagnosis'] == dx).sum()
                logger.info(f"  {dx}: {count}")

        logger.info("="*80 + "\n")

    def extract_subject_regional_features(self,
                                         participant_id: str,
                                         sess_id: str,
                                         rid: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Extract regional features for a single subject.

        Args:
            participant_id: Subject ID (e.g., 'sub-ADNIAAASBBBB')
            sess_id: Session ID (e.g., 'ses-M132')
            rid: RID number (optional, for additional matching)

        Returns:
            Dictionary with arrays of shape (112,) for each feature type:
                'smri_volume_z': Volume z-scores per region
                'smri_thickness_z': Thickness z-scores per region
                'tau_suvr': TAU SUVR per region
                'amyloid_suvr': Amyloid SUVR per region
        """

        # Look up subject in regional PET data
        subject_mask = (
            (self.regional_pet_df['participant_id'] == participant_id) &
            (self.regional_pet_df['session_id'] == sess_id)
        )

        subject_data = self.regional_pet_df[subject_mask]

        if len(subject_data) == 0:
            logger.debug(f"No regional data for {participant_id} {sess_id}")
            return self._get_default_features()

        if len(subject_data) > 1:
            logger.warning(f"Multiple matches for {participant_id} {sess_id}, using first")

        subject_row = subject_data.iloc[0]

        # Initialize feature arrays for 112 DK nodes
        tau_suvr = np.zeros(112, dtype=np.float32)
        amyloid_suvr = np.zeros(112, dtype=np.float32)
        smri_volume = np.zeros(112, dtype=np.float32)
        smri_thickness = np.zeros(112, dtype=np.float32)

        # Extract PET values, depending on mode
        regions_with_tau = 0
        regions_with_amyloid = 0

        if REGIONAL_MODE == "broadcast_pet_regional_smri":
            # --- APPLES-TO-APPLES: use global PET only, broadcast to cortical DK nodes (0..67) ---
            tau_global = None
            amy_global = None

            # Tau global (use META_TEMPORAL_SUVR if available, else CTX_ENTORHINAL_SUVR)
            for col in ["META_TEMPORAL_SUVR", "CTX_ENTORHINAL_SUVR"]:
                if col in subject_row.index and pd.notna(subject_row[col]):
                    tau_global = float(subject_row[col])
                    break

            # Amyloid global (use SUMMARY_SUVR if available; fallback to AMYLOID_HIPPOCAMPUS_SUVR rarely)
            for col in ["SUMMARY_SUVR", "AMYLOID_HIPPOCAMPUS_SUVR"]:
                if col in subject_row.index and pd.notna(subject_row[col]):
                    amy_global = float(subject_row[col])
                    break

            # Broadcast to cortical DK nodes only (0..67); leave subcortical (68..111) unchanged
            if tau_global is not None:
                tau_suvr[:68] = tau_global
                regions_with_tau = 68
            if amy_global is not None:
                amyloid_suvr[:68] = amy_global
                regions_with_amyloid = 68

        else:
            # --- FULL REGIONAL PET: LH/RH with bilateral fallback into each cortical DK node ---
            for region_name, node_idx in self.region_mapping.items():
                if node_idx >= 112:
                    continue  # we only fill cortical DK here

                # region_name like CTX_LH_FUSIFORM, CTX_RH_FUSIFORM
                if region_name.startswith("CTX_LH_"):
                    bilateral = "CTX_" + region_name[len("CTX_LH_"):]
                elif region_name.startswith("CTX_RH_"):
                    bilateral = "CTX_" + region_name[len("CTX_RH_"):]
                else:
                    bilateral = region_name  # subcortical or already bilateral-ish

                # TAU hemi -> bilateral fallback
                tau_col_hemi = f"TAU_{region_name}_SUVR"
                tau_col_bi   = f"TAU_{bilateral}_SUVR"
                tau_val = None
                if tau_col_hemi in subject_row.index and pd.notna(subject_row[tau_col_hemi]):
                    tau_val = subject_row[tau_col_hemi]
                elif tau_col_bi in subject_row.index and pd.notna(subject_row[tau_col_bi]):
                    tau_val = subject_row[tau_col_bi]
                if tau_val is not None:
                    tau_suvr[node_idx] = float(tau_val)
                    regions_with_tau += 1

                # AMY hemi -> bilateral fallback
                amy_col_hemi = f"AMYLOID_{region_name}_SUVR"
                amy_col_bi   = f"AMYLOID_{bilateral}_SUVR"
                amy_val = None
                if amy_col_hemi in subject_row.index and pd.notna(subject_row[amy_col_hemi]):
                    amy_val = subject_row[amy_col_hemi]
                elif amy_col_bi in subject_row.index and pd.notna(subject_row[amy_col_bi]):
                    amy_val = subject_row[amy_col_bi]
                if amy_val is not None:
                    amyloid_suvr[node_idx] = float(amy_val)
                    regions_with_amyloid += 1

        # Log coverage for this subject
        if regions_with_tau > 0 or regions_with_amyloid > 0:
            logger.debug(
                f"{participant_id} {sess_id}: "
                f"TAU={regions_with_tau}/112, "
                f"Amyloid={regions_with_amyloid}/112"
            )


        # Prefer TRUE regional sMRI if available; otherwise fall back to proxy/global
        if self.regional_smri_df is not None:
            df = self.regional_smri_df

            # build the mask
            mask = df['participant_id'].eq(participant_id)

            # Only constrain by session if the dataframe actually has valid session_id values
            use_session = ('session_id' in df.columns) and df['session_id'].notna().any() and (sess_id is not None)
            if use_session:
                mask &= df['session_id'].eq(sess_id)

            # use the SAME variable name you built: `mask`
            if mask.any():
                smri_row = df.loc[mask].iloc[0]
        # Prefer TRUE regional sMRI if available; otherwise fall back to proxy/global
        if self.regional_smri_df is not None:
            df = self.regional_smri_df

            # build the mask
            mask = df['participant_id'].eq(participant_id)

            # Only constrain by session if the dataframe actually has valid session_id values
            use_session = ('session_id' in df.columns) and df['session_id'].notna().any() and (sess_id is not None)
            if use_session:
                mask &= df['session_id'].eq(sess_id)

            if ST_MAPPING_AVAILABLE and mask.any():
                smri_row = df.loc[mask].iloc[0]

                # ---- Fill sMRI per DK node from UCSFFSX7 STxx{CV,TA,SA} ----
                # case-insensitive getter (handles oddities like 'ST87sa')
                def _get_ci(row: pd.Series, key: str, default=np.nan):
                    key_u = key.upper()
                    for k in row.index:
                        if str(k).upper() == key_u:
                            return row[k]
                    return default

                # Left hemisphere DK -> indices 0..33
                for i, lab in enumerate(DK_34):
                    st = DK2ST_L[lab]
                    cv = _get_ci(smri_row, f"ST{st}CV")  # Cortical Volume
                    ta = _get_ci(smri_row, f"ST{st}TA")  # Thickness Average
                    if pd.notna(cv):
                        smri_volume[i] = float(cv)
                    if pd.notna(ta):
                        smri_thickness[i] = float(ta)

                # Right hemisphere DK -> indices 34..67
                for j, lab in enumerate(DK_34, start=34):
                    st = DK2ST_R[lab]
                    cv = _get_ci(smri_row, f"ST{st}CV")
                    ta = _get_ci(smri_row, f"ST{st}TA")
                    if pd.notna(cv):
                        smri_volume[j] = float(cv)
                    if pd.notna(ta):
                        smri_thickness[j] = float(ta)

            else:
                # Fallback to proxy/global sMRI features when no UCSFFSX7 match exists
                smri_volume, smri_thickness = self._extract_smri_features(
                    subject_row, smri_volume, smri_thickness
                )

        # Z-score normalization
        tau_suvr = self._normalize_regional_features(tau_suvr, 'tau')
        amyloid_suvr = self._normalize_regional_features(amyloid_suvr, 'amyloid')

        return {
            'smri_volume_z': smri_volume,
            'smri_thickness_z': smri_thickness,
            'tau_suvr': tau_suvr,
            'amyloid_suvr': amyloid_suvr
        }

    def _extract_smri_features(self,
                              subject_row: pd.Series,
                              smri_volume: np.ndarray,
                              smri_thickness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract sMRI features and assign to relevant brain regions.

        """

        # Hippocampus (nodes 68-69 in DK)
        hippo_val = 0.0
        if 'SMRI_HIPPOCAMPUS_BILATERAL' in subject_row.index:
            hippo_val = subject_row['SMRI_HIPPOCAMPUS_BILATERAL']
            if pd.notna(hippo_val):
                # Normalize
                if 'SMRI_HIPPOCAMPUS_BILATERAL' in self.cn_stats:
                    stat = self.cn_stats['SMRI_HIPPOCAMPUS_BILATERAL']
                    hippo_val = (float(hippo_val) - stat['mean']) / stat['std']

                    # Assign to hippocampal nodes
                    for idx in [68, 69]:  # LEFT/RIGHT_HIPPOCAMPUS
                        if idx < 112:
                            smri_volume[idx] = hippo_val

        # Entorhinal (nodes 4, 38 in DK)
        entorh_val = 0.0
        if 'SMRI_ENTORHINAL_BILATERAL' in subject_row.index:
            entorh_val = subject_row['SMRI_ENTORHINAL_BILATERAL']
            if pd.notna(entorh_val):
                if 'SMRI_ENTORHINAL_BILATERAL' in self.cn_stats:
                    stat = self.cn_stats['SMRI_ENTORHINAL_BILATERAL']
                    entorh_val = (float(entorh_val) - stat['mean']) / stat['std']

                    # Assign to entorhinal nodes
                    for idx in [4, 38]:  # CTX_LH/RH_ENTORHINAL
                        if idx < 112:
                            smri_thickness[idx] = entorh_val

        # Amygdala (nodes 70-71 in DK)
        amyg_val = 0.0
        if 'SMRI_AMYGDALA_BILATERAL' in subject_row.index:
            amyg_val = subject_row['SMRI_AMYGDALA_BILATERAL']
            if pd.notna(amyg_val):
                if 'SMRI_AMYGDALA_BILATERAL' in self.cn_stats:
                    stat = self.cn_stats['SMRI_AMYGDALA_BILATERAL']
                    amyg_val = (float(amyg_val) - stat['mean']) / stat['std']

                    # Assign to amygdala nodes
                    for idx in [70, 71]:  # LEFT/RIGHT_AMYGDALA
                        if idx < 112:
                            smri_volume[idx] = amyg_val

        # Global gray matter (assign to all cortical nodes)
        if 'SMRI_TOTAL_GRAY_MATTER' in subject_row.index:
            gm_val = subject_row['SMRI_TOTAL_GRAY_MATTER']
            if pd.notna(gm_val) and 'SMRI_TOTAL_GRAY_MATTER' in self.cn_stats:
                stat = self.cn_stats['SMRI_TOTAL_GRAY_MATTER']
                gm_val_z = (float(gm_val) - stat['mean']) / stat['std']

                # Assign to cortical nodes (0-67) where no specific value exists
                for idx in range(68):
                    if smri_volume[idx] == 0.0:  # Not already assigned
                        smri_volume[idx] = gm_val_z * 0.5  # Scaled down

        return smri_volume, smri_thickness

    def _normalize_regional_features(self,
                                    features: np.ndarray,
                                    feature_type: str) -> np.ndarray:
        """
        Normalize regional features using CN statistics.

        Args:
            features: Array of shape (112,) with regional values
            feature_type: 'tau' or 'amyloid'

        Returns:
            Normalized array
        """
        normalized = features.copy()

        for node_idx in range(112):
            if features[node_idx] != 0.0:  # Has data
                stat_key = f'{feature_type}_{node_idx}'
                if stat_key in self.cn_stats:
                    stat = self.cn_stats[stat_key]
                    normalized[node_idx] = (features[node_idx] - stat['mean']) / stat['std']

        return normalized

    def _get_default_features(self) -> Dict[str, np.ndarray]:
        """Return default (zero) features when data is missing"""
        return {
            'smri_volume_z': np.zeros(112, dtype=np.float32),
            'smri_thickness_z': np.zeros(112, dtype=np.float32),
            'tau_suvr': np.zeros(112, dtype=np.float32),
            'amyloid_suvr': np.zeros(112, dtype=np.float32)
        }


def compute_node_features_dk_regional(
    fc_tensor: torch.Tensor,
    regional_features: Dict[str, np.ndarray],
    age_z: float,
    sex_bin: float,
    apoe2_dos: Tuple[float, float, float],
    apoe4_dos: Tuple[float, float, float]
) -> torch.Tensor:
    """
    Compute per-node features for DK parcellation with regional biomarkers.

    Feature structure (19 dimensions per node):
        [0-2]   Connectivity: strength, degree_norm, clustering
        [3-4]   Demographics: age_z, sex_bin
        [5-10]  APOE: e2_0, e2_1, e2_2, e4_0, e4_1, e4_2
        [11]    TAU SUVR (regional)
        [12]    Amyloid SUVR (regional)
        [13]    sMRI volume z-score (regional)
        [14]    sMRI thickness z-score (regional)
        [15]    TAU positive (binary, threshold > 1.3)
        [16]    Amyloid positive (binary, threshold > 1.11)
        [17]    Volume atrophy (binary, z < -1.5)
        [18]    Thickness atrophy (binary, z < -1.5)

    Args:
        fc_tensor: Functional connectivity matrix [N, N]
        regional_features: Dictionary with regional biomarker arrays
        age_z: Age z-score
        sex_bin: Sex binary (1=M, 0=F)
        apoe2_dos: APOE2 dosage (3-element tuple)
        apoe4_dos: APOE4 dosage (3-element tuple)

    Returns:
        Node features tensor of shape [N, 19]
    """
    N = fc_tensor.shape[0]
    device = fc_tensor.device

    with torch.no_grad():
        # 1. Connectivity features [0-2]
        W = fc_tensor.abs().clone()
        W.fill_diagonal_(0.0)

        strength = W.mean(dim=1)
        deg = W.sum(dim=1)
        dmin, dmax = deg.min(), deg.max()
        degree_norm = (deg - dmin) / (dmax - dmin + 1e-6)

        # Clustering coefficient
        W13 = torch.pow(W, 1.0/3.0)
        T = W13 @ W13 @ W13
        k = (W > 0).sum(dim=1).float()
        denom = k * (k - 1.0)
        clust = torch.zeros(N, dtype=torch.float32, device=device)
        m = denom > 0
        clust[m] = torch.diag(T)[m] / denom[m]

        # 2. Demographics (broadcast to all nodes) [3-4]
        age_feat = torch.full((N,), age_z, dtype=torch.float32, device=device)
        sex_feat = torch.full((N,), sex_bin, dtype=torch.float32, device=device)

        # 3. APOE (broadcast to all nodes) [5-10]
        apoe2_0, apoe2_1, apoe2_2 = apoe2_dos
        apoe4_0, apoe4_1, apoe4_2 = apoe4_dos
        e2_0 = torch.full((N,), apoe2_0, dtype=torch.float32, device=device)
        e2_1 = torch.full((N,), apoe2_1, dtype=torch.float32, device=device)
        e2_2 = torch.full((N,), apoe2_2, dtype=torch.float32, device=device)
        e4_0 = torch.full((N,), apoe4_0, dtype=torch.float32, device=device)
        e4_1 = torch.full((N,), apoe4_1, dtype=torch.float32, device=device)
        e4_2 = torch.full((N,), apoe4_2, dtype=torch.float32, device=device)

        # 4. Regional biomarkers - pad or trim to match N [11-14]
        tau_suvr = np.zeros(N)
        amy_suvr = np.zeros(N)
        vol_z = np.zeros(N)
        thick_z = np.zeros(N)

        # Copy available data (up to N nodes)
        n_features = min(N, len(regional_features['tau_suvr']))
        tau_suvr[:n_features] = regional_features['tau_suvr'][:n_features]
        amy_suvr[:n_features] = regional_features['amyloid_suvr'][:n_features]
        vol_z[:n_features] = regional_features['smri_volume_z'][:n_features]
        thick_z[:n_features] = regional_features['smri_thickness_z'][:n_features]

        # Convert to tensors
        tau_suvr_t = torch.tensor(tau_suvr, dtype=torch.float32, device=device)
        amy_suvr_t = torch.tensor(amy_suvr, dtype=torch.float32, device=device)
        vol_z_t = torch.tensor(vol_z, dtype=torch.float32, device=device)
        thick_z_t = torch.tensor(thick_z, dtype=torch.float32, device=device)

        # 5. Binary indicators [15-18]
        tau_pos = (tau_suvr_t > 1.3).float()
        amy_pos = (amy_suvr_t > 1.11).float()
        vol_atrophy = (vol_z_t < -1.5).float()
        thick_atrophy = (thick_z_t < -1.5).float()

        # Stack all features [N, 19]
        x = torch.stack([
            strength, degree_norm, clust,  # 0-2: connectivity
            age_feat, sex_feat,  # 3-4: demographics
            e2_0, e2_1, e2_2, e4_0, e4_1, e4_2,  # 5-10: APOE
            tau_suvr_t, amy_suvr_t, vol_z_t, thick_z_t,  # 11-14: regional continuous
            tau_pos, amy_pos, vol_atrophy, thick_atrophy  # 15-18: binary indicators
        ], dim=1)

    return x  # [N, 19]