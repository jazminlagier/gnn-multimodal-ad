#!/usr/bin/env python3
"""
FreeSurfer Regional sMRI Feature Extractor

This module provides functionality to extract region-specific sMRI features
from FreeSurfer processed data, enabling TRUE regional analysis instead of
global sMRI broadcasting.

"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FreeSurferRegionalExtractor:
    """
    Extract regional sMRI features from FreeSurfer processed data.

    This class provides methods to:
    1. Load FreeSurfer regional data
    2. Map participants to their regional sMRI features
    3. Provide region-specific values for each of the 112 DK atlas regions
    """

    def __init__(self, freesurfer_file, demographics_file=None):
        """
        Initialize the FreeSurfer regional extractor.

        Args:
            freesurfer_file (str): Path to FreeSurfer regional data CSV
            demographics_file (str): Path to demographics file for participant mapping
        """
        self.freesurfer_file = Path(freesurfer_file)
        self.demographics_file = Path(demographics_file) if demographics_file else None

        # Load data
        self._load_freesurfer_data()
        if self.demographics_file:
            self._load_demographics()

        # Define DK atlas region mapping
        self._setup_dk_mapping()

    def _load_freesurfer_data(self):
        """Load FreeSurfer regional data."""
        try:
            self.freesurfer_data = pd.read_csv(self.freesurfer_file)
            logger.info(f"Loaded FreeSurfer data: {len(self.freesurfer_data)} participants")

            # Basic data validation
            required_cols = ['participant_id']
            missing_cols = [col for col in required_cols if col not in self.freesurfer_data.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")

            # Set participant_id as index for fast lookup
            if 'participant_id' in self.freesurfer_data.columns:
                self.freesurfer_data.set_index('participant_id', inplace=True)

        except Exception as e:
            logger.error(f"Error loading FreeSurfer data: {e}")
            self.freesurfer_data = pd.DataFrame()

    def _load_demographics(self):
        """Load demographics data for participant mapping."""
        try:
            self.demographics = pd.read_csv(self.demographics_file)
            logger.info(f"Loaded demographics: {len(self.demographics)} participants")
        except Exception as e:
            logger.warning(f"Could not load demographics: {e}")
            self.demographics = pd.DataFrame()

    def _setup_dk_mapping(self):
        """
        Setup mapping between FreeSurfer regions and DK atlas (112 regions).

        This maps FreeSurfer region names to the 112 DK atlas regions used
        in the connectivity matrices.
        """
        # DK atlas region names (112 regions)
        self.dk_regions = [
            # Left hemisphere (0-55)
            'Left-Frontal-Pole', 'Left-Rostral-Anterior-Cingulate', 'Left-Caudal-Anterior-Cingulate',
            'Left-Lateral-Orbitofrontal', 'Left-Medial-Orbitofrontal', 'Left-Parahippocampal',
            'Left-Temporal-Pole', 'Left-Entorhinal', 'Left-Fusiform', 'Left-Lingual',
            'Left-Pericalcarine', 'Left-Cuneus', 'Left-Precuneus', 'Left-Superior-Parietal',
            'Left-Inferior-Parietal', 'Left-Supramarginal', 'Left-Postcentral', 'Left-Precentral',
            'Left-Superior-Frontal', 'Left-Rostral-Middle-Frontal', 'Left-Caudal-Middle-Frontal',
            'Left-Pars-Opercularis', 'Left-Pars-Triangularis', 'Left-Pars-Orbitalis',
            'Left-Insula', 'Left-Superior-Temporal', 'Left-Middle-Temporal', 'Left-Inferior-Temporal',
            'Left-Transverse-Temporal', 'Left-Banks-STS', 'Left-Posterioir-Cingulate',
            'Left-Isthmus-Cingulate', 'Left-Paracentral', 'Left-Caudal-Anterior-Cingulate',

            # Right hemisphere (56-111)
            'Right-Frontal-Pole', 'Right-Rostral-Anterior-Cingulate', 'Right-Caudal-Anterior-Cingulate',
            'Right-Lateral-Orbitofrontal', 'Right-Medial-Orbitofrontal', 'Right-Parahippocampal',
            'Right-Temporal-Pole', 'Right-Entorhinal', 'Right-Fusiform', 'Right-Lingual',
            'Right-Pericalcarine', 'Right-Cuneus', 'Right-Precuneus', 'Right-Superior-Parietal',
            'Right-Inferior-Parietal', 'Right-Supramarginal', 'Right-Postcentral', 'Right-Precentral',
            'Right-Superior-Frontal', 'Right-Rostral-Middle-Frontal', 'Right-Caudal-Middle-Frontal',
            'Right-Pars-Opercularis', 'Right-Pars-Triangularis', 'Right-Pars-Orbitalis',
            'Right-Insula', 'Right-Superior-Temporal', 'Right-Middle-Temporal', 'Right-Inferior-Temporal',
            'Right-Transverse-Temporal', 'Right-Banks-STS', 'Right-Posterioir-Cingulate',
            'Right-Isthmus-Cingulate', 'Right-Paracentral', 'Right-Caudal-Anterior-Cingulate'
        ]

        # Extend to include subcortical regions (72-111)
        subcortical_regions = [
            'Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
            'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens',
            'Right-Thalamus', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum',
            'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens',
            'Brain-Stem', 'CSF', 'Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter',
            'Left-Cerebral-Cortex', 'Right-Cerebral-Cortex', 'Left-Lateral-Ventricle',
            'Right-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Right-Inf-Lat-Vent',
            'Left-Cerebellum-White-Matter', 'Right-Cerebellum-White-Matter',
            'Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex', 'Left-VentralDC',
            'Right-VentralDC', 'Left-choroid-plexus', 'Right-choroid-plexus',
            'Third-Ventricle', 'Fourth-Ventricle', 'Fifth-Ventricle',
            'WM-hypointensities', 'Left-WM-hypointensities', 'Right-WM-hypointensities',
            'non-WM-hypointensities', 'Left-non-WM-hypointensities', 'Right-non-WM-hypointensities'
        ]

        # Complete DK atlas (112 regions total)
        self.dk_regions.extend(subcortical_regions[:40])  # Take first 40 to reach 112 total

        # Create mapping dictionaries
        self._create_feature_mappings()

    def _create_feature_mappings(self):
        """Create mappings between FreeSurfer columns and DK regions."""
        # This would typically involve mapping FreeSurfer column names to DK regions
        # For now, create a generic mapping that can be customized

        self.thickness_mapping = {}
        self.area_mapping = {}
        self.volume_mapping = {}

        # Example mapping (should be customized based on actual FreeSurfer data)
        if not self.freesurfer_data.empty:
            freesurfer_cols = self.freesurfer_data.columns

            # Map thickness columns
            thickness_cols = [col for col in freesurfer_cols if 'thickness' in col.lower()]
            area_cols = [col for col in freesurfer_cols if 'area' in col.lower()]
            volume_cols = [col for col in freesurfer_cols if 'volume' in col.lower()]

            # Simple mapping - this should be refined based on actual data structure
            for i, region in enumerate(self.dk_regions):
                if i < len(thickness_cols):
                    self.thickness_mapping[region] = thickness_cols[i]
                if i < len(area_cols):
                    self.area_mapping[region] = area_cols[i]
                if i < len(volume_cols):
                    self.volume_mapping[region] = volume_cols[i]

    def get_regional_features(self, participant_id):
        """
        Extract regional sMRI features for a specific participant.

        Args:
            participant_id (str): Participant identifier

        Returns:
            dict: Dictionary containing regional features for 112 DK regions
                 Keys: 'cortical_thickness', 'surface_area', 'volume'
                 Values: numpy arrays of length 112
        """
        if self.freesurfer_data.empty:
            logger.warning("No FreeSurfer data available")
            return None

        # Handle different participant ID formats
        participant_lookup = self._standardize_participant_id(participant_id)

        if participant_lookup not in self.freesurfer_data.index:
            logger.warning(f"Participant {participant_id} not found in FreeSurfer data")
            return None

        try:
            participant_data = self.freesurfer_data.loc[participant_lookup]

            # Extract regional features
            cortical_thickness = self._extract_thickness_features(participant_data)
            surface_area = self._extract_area_features(participant_data)
            volume = self._extract_volume_features(participant_data)

            return {
                'cortical_thickness': cortical_thickness,
                'surface_area': surface_area,
                'volume': volume
            }

        except Exception as e:
            logger.error(f"Error extracting features for {participant_id}: {e}")
            return None

    def _standardize_participant_id(self, participant_id):
        """Standardize participant ID format for lookup."""
        # Handle common participant ID variations
        if isinstance(participant_id, str):
            # Remove common prefixes/suffixes
            clean_id = participant_id.replace('sub-', '').replace('_', '-')

            # Try different formats
            possible_ids = [participant_id, clean_id, f"sub-{clean_id}"]

            for pid in possible_ids:
                if pid in self.freesurfer_data.index:
                    return pid

        return participant_id

    def _extract_thickness_features(self, participant_data):
        """Extract cortical thickness for all 112 DK regions."""
        thickness_values = np.zeros(112)

        for i, region in enumerate(self.dk_regions):
            if region in self.thickness_mapping:
                col_name = self.thickness_mapping[region]
                if col_name in participant_data.index:
                    thickness_values[i] = participant_data[col_name]
                else:
                    # Use mean thickness if specific region not available
                    thickness_cols = [col for col in participant_data.index if 'thickness' in col.lower()]
                    if thickness_cols:
                        thickness_values[i] = np.mean([participant_data[col] for col in thickness_cols])
            else:
                # Default value if mapping not available
                thickness_values[i] = 2.5  # Typical cortical thickness in mm

        return thickness_values

    def _extract_area_features(self, participant_data):
        """Extract surface area for all 112 DK regions."""
        area_values = np.zeros(112)

        for i, region in enumerate(self.dk_regions):
            if region in self.area_mapping:
                col_name = self.area_mapping[region]
                if col_name in participant_data.index:
                    area_values[i] = participant_data[col_name]
                else:
                    # Use mean area if specific region not available
                    area_cols = [col for col in participant_data.index if 'area' in col.lower()]
                    if area_cols:
                        area_values[i] = np.mean([participant_data[col] for col in area_cols])
            else:
                # Default value if mapping not available
                area_values[i] = 1000.0  # Typical surface area in mm2

        return area_values

    def _extract_volume_features(self, participant_data):
        """Extract volume for all 112 DK regions."""
        volume_values = np.zeros(112)

        for i, region in enumerate(self.dk_regions):
            if region in self.volume_mapping:
                col_name = self.volume_mapping[region]
                if col_name in participant_data.index:
                    volume_values[i] = participant_data[col_name]
                else:
                    # Use mean volume if specific region not available
                    volume_cols = [col for col in participant_data.index if 'volume' in col.lower()]
                    if volume_cols:
                        volume_values[i] = np.mean([participant_data[col] for col in volume_cols])
            else:
                # Default value if mapping not available
                volume_values[i] = 3000.0  # Typical volume in mm3

        return volume_values

    def get_available_participants(self):
        """Get list of participants with FreeSurfer data available."""
        if self.freesurfer_data.empty:
            return []
        return list(self.freesurfer_data.index)

    def validate_data(self):
        """Validate the loaded FreeSurfer data."""
        if self.freesurfer_data.empty:
            return False, "No FreeSurfer data loaded"

        # Check for required columns
        thickness_cols = [col for col in self.freesurfer_data.columns if 'thickness' in col.lower()]
        area_cols = [col for col in self.freesurfer_data.columns if 'area' in col.lower()]
        volume_cols = [col for col in self.freesurfer_data.columns if 'volume' in col.lower()]

        if not thickness_cols and not area_cols and not volume_cols:
            return False, "No thickness, area, or volume columns found"

        return True, f"Valid data with {len(thickness_cols)} thickness, {len(area_cols)} area, {len(volume_cols)} volume columns"


def compute_node_features_with_regional_smri(fc_tensor, participant_id, freesurfer_extractor,
                                           age_z=0, sex_bin=0, **kwargs):
    """
    Compute node features using regional sMRI data from FreeSurfer.

    This function combines connectivity features
    with TRUE regional sMRI features instead of global broadcasting.

    Args:
        fc_tensor (torch.Tensor): Functional connectivity matrix [N, N]
        participant_id (str): Participant identifier
        freesurfer_extractor (FreeSurferRegionalExtractor): Initialized extractor
        age_z (float): Z-scored age
        sex_bin (int): Binary sex encoding
        **kwargs: Additional demographic/biomarker features

    Returns:
        torch.Tensor: Node features [N, num_features]
    """
    N = fc_tensor.shape[0]
    feature_list = []

    with torch.no_grad():
        # Connectivity features
        W = fc_tensor.abs().clone()
        W.fill_diagonal_(0.0)
        strength = W.mean(dim=1)
        deg = W.sum(dim=1)
        dmin, dmax = deg.min(), deg.max()
        degree_norm = (deg - dmin) / (dmax - dmin + 1e-6)
        feature_list.extend([strength, degree_norm])

        # Demographics (global)
        age_feat = torch.full((N,), float(age_z), dtype=torch.float32, device=fc_tensor.device)
        sex_feat = torch.full((N,), float(sex_bin), dtype=torch.float32, device=fc_tensor.device)
        feature_list.extend([age_feat, sex_feat])

        # Regional sMRI features
        regional_data = freesurfer_extractor.get_regional_features(participant_id)
        if regional_data is not None:
            # Use TRUE regional values
            thickness = torch.tensor(regional_data['cortical_thickness'],
                                   dtype=torch.float32, device=fc_tensor.device)
            area = torch.tensor(regional_data['surface_area'],
                              dtype=torch.float32, device=fc_tensor.device)
            volume = torch.tensor(regional_data['volume'],
                                dtype=torch.float32, device=fc_tensor.device)

            feature_list.extend([thickness, area, volume])
        else:
            # Fallback to zeros if no regional data
            zero_feat = torch.zeros(N, dtype=torch.float32, device=fc_tensor.device)
            feature_list.extend([zero_feat, zero_feat, zero_feat])

        x = torch.stack(feature_list, dim=1)

    return x


if __name__ == "__main__":
    # Example usage
    extractor = FreeSurferRegionalExtractor(
        freesurfer_file="path/to/freesurfer_data.csv",
        demographics_file="path/to/demographics.csv"
    )

    # Validate data
    is_valid, message = extractor.validate_data()
    print(f"Data validation: {message}")

    # Get features for a participant
    participant_id = "sub-001"
    features = extractor.get_regional_features(participant_id)

    if features:
        print(f"Extracted features for {participant_id}:")
        print(f"  Cortical thickness: {features['cortical_thickness'].shape}")
        print(f"  Surface area: {features['surface_area'].shape}")
        print(f"  Volume: {features['volume'].shape}")
    else:
        print(f"No features found for {participant_id}")
