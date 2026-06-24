# Data Access

This project uses data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). ADNI data are not redistributed in this repository.

## Obtaining the data

Access must be requested directly through the ADNI/LONI Image and Data Archive. Users are responsible for reviewing and complying with the ADNI Data Use Agreement before downloading or using any ADNI data.

## Expected processed inputs

The runner expects processed local files whose paths are set in `configs/example_config.yaml`:

- `fc_matrices_dir`: per-session Desikan-Killiany functional connectivity matrices.
- `demographics_file`: age, sex, APOE genotype, diagnosis, participant/session identifiers.
- `regional_pet_file`: per-session regional tau and amyloid PET summaries.
- `regional_smri_file` or `freesurfer_file` optional regional structural MRI features.

The raw ADNI data, preprocessing scripts specific to local protected storage, and participant/session-level derived data are not included.

## Sharing restrictions used for this repository

The following are intentionally excluded:

- raw or processed participant-level ADNI data;
- functional connectivity matrices;
- regional PET or sMRI tables;
- APOE/demographic tables;
- RID, PTID, VISCODE, or other participant/session-level outputs;
- trained model checkpoints derived from ADNI data;
- per-participant predictions, attention weights, or interpretability files;
- local configuration files with protected paths.

This repository is intended to share the code and workflow only. Results should be reported in the manuscript or shared only as aggregate, non-identifiable summaries when allowed by the applicable data-use agreement.
