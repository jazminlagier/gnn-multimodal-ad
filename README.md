# GNN Multimodal AD

Code for multimodal graph neural network classification of Alzheimer's disease using ADNI-derived neuroimaging and clinical features.

The project represents each session as a brain graph over the Desikan-Killiany atlas. Resting-state fMRI functional connectivity defines graph edges, while node features can include regional PET, structural MRI, APOE genotype, age, and sex. The default configuration is set up for the GAT model used in the manuscript, with GCN and GKAN implementations retained as optional comparison models.

## Repository contents

```text
.
├── src/
│   ├── main_gnn_adni.py
│   ├── data_utils_v25.py
│   ├── baseline_models_v25.py
│   ├── train_utils_v25.py
│   ├── dk_interpretability.py
│   ├── dk_region_mapping.py
│   ├── regional_features_dk.py
│   ├── regional_smri_freesurfer_extractor.py
│   ├── learning_curve_tracker.py
│   ├── gkan_simple_v25.py
│   └── kan_layers_v25.py
├── configs/
│   └── example_config.yaml
├── docs/
│   └── DATA_ACCESS.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## What is intentionally not included

ADNI data are not redistributed in this repository. The repository does not include raw imaging data, functional connectivity matrices, regional PET tables, regional sMRI tables, APOE/demographic tables, participant/session-level predictions, trained checkpoints, or per-participant interpretability outputs.

Results are reported in the associated manuscript. Only source code, documentation, and configuration templates are included here.

## Installation

```bash
git clone https://github.com/<your-username>/gnn-multimodal-ad.git
cd gnn-multimodal-ad
pip install -r requirements.txt
```

Install PyTorch, PyTorch Geometric, and related packages with versions matching your CUDA environment.

## Configuration

Copy the example configuration and edit the paths:

```bash
cp configs/example_config.yaml configs/my_config.yaml
```

Then replace all `/path/to/...` placeholders with local paths to the processed ADNI-derived files available under your approved ADNI access.

## Running the model

```bash
python src/main_gnn_adni.py --config configs/my_config.yaml
```

By default, `configs/example_config.yaml` runs the GAT model only. To run comparison models, edit:

```yaml
models:
  - gat
  - gcn
  - gkan
```

## Data access

See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md). Users must obtain ADNI data directly from the official ADNI/LONI access process and comply with the ADNI Data Use Agreement.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
