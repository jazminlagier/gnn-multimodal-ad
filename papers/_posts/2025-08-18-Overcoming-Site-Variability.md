---
layout: paper
title: "Overcoming Site Variability in Multisite fMRI Studies: An Autoencoder Framework for Enhanced Generalizability of Machine Learning Models"
nickname: AE-Harmonization-paper
authors: "Almuqhim, Fahad; Saeed, Fahad; "
year: "2025"
journal: "Springer Neuroinformatics"
volume: 23
issue: article 46
pages:
is_published: True
image: /assets/images/papers/neuroinformatics.jpeg
projects: [ML-brain-imaging]
tags: [ML, AE, ComBat, ASD, Multisite]

# Text
fulltext: https://link.springer.com/article/10.1007/s12021-025-09746-1
pdf:
pdflink: https://link.springer.com/content/pdf/10.1007/s12021-025-09746-1.pdf
pmcid:
preprint:  
supplement:

# Links
doi: 10.1007/s12021-025-09746-1
pmid:

# Data and code
github: [https://github.com/pcdslab/Autoencoder-fMRI-Harmonization]
neurovault:
openneuro: []
figshare:
figshare_names:
osf: [d8253]
---
{% include JB/setup %}

# Abstract

Harmonizing multisite functional magnetic resonance imaging (fMRI) data is crucial for eliminating site-specific variability that hinders the generalizability of machine learning models. Traditional harmonization techniques, such as ComBat, depend on additive and multiplicative factors, and may struggle to capture the non-linear interactions between scanner hardware, acquisition protocols, and signal variations between different imaging sites. In addition, these statistical techniques require data from all the sites during their model training which may have the unintended consequence of data leakage for ML models trained using this harmonized data. The ML models trained using this harmonized data may result in low reliability and reproducibility when tested on unseen data sets, limiting their applicability for general clinical usage. In this study, we propose Autoencoders (AEs) as an alternative for harmonizing multisite fMRI data. Our designed and developed framework leverages the non-linear representation learning capabilities of AEs to reduce site-specific effects while preserving biologically meaningful features. Our evaluation using Autism Brain Imaging Data Exchange I (ABIDE-I) dataset, containing 1,035 subjects collected from 17 centers demonstrates statistically significant improvements in leave-one-site-out (LOSO) cross-validation evaluations. All AE variants (AE, SAE, TAE, and DAE) significantly outperformed the baseline mode (p<0.01), with mean accuracy improvements ranging from 3.41% to 5.04%. Our findings demonstrate the potential of AEs to harmonize multisite neuroimaging data effectively enabling robust downstream analyses across various neuroscience applications while reducing data-leakage, and preservation of neurobiological features. Our open-source code is made available at https://github.com/pcdslab/Autoencoder-fMRI-Harmonization