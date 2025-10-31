---
layout: paper
title: "Beyond Mass-Only Filtering: A Multitask Deep Learning Framework for Predicting Peptide Properties from Mass Spectrometry Data"
nickname: asms-massspectrometry-2025-poster
authors: "Bilal Shabbir; Usman Tariq; Syed Talha Khalid; Francisco Fernandez-Lima; Fahad Saeed"
year: "2025"
conference: "ASMS-2025"
image: /assets/images/posters/beyondMOF-ASMS-2025-poster.png
projects: ["ML-MS"]
tags: []

# Content
# fulltext: https://alz.confex.com/alz/2024/meetingapp.cgi/Paper/89944
pdf: 

# Links
doi: 


# Data and code
github:
neurovault:
openneuro:
figshare:
figshare_names:
osf:
f1000:
---

{% include JB/setup %}

# Introduction
Mass-based filtering is widely used to improve scalability in shotgun proteomics database searches. However, the usage of these mass-filters may result in missing potential modifications, chemical reactions, instrumental artifacts such as incorrect isotope selection, and labile modifications. Spectra and peptide properties can be utilized for filtering to perform a more refined and scalable search. To address these challenges, we propose Proteorift, a deep-learning multitask model that predicts peptide features, including length, missed cleavages, and modifications, directly from the spectra. These predictions can guide more effective filtering and improved peptide deductions. Our proposed approach achieves 12x speedup which highlight the significance of applying filter beyond mass.


# Methods
The methods employed in this study involve several key steps. First, embeddings were generated using a deep attention-based network to create spectra and peptide representations. A multi-task network was then utilized to predict peptide properties, including length, missed cleavages, and modification status, directly from the spectra. Spectra and peptides were binned into batches based on these predictions to optimize the search process. Peptide deduction was performed using batching and predicted properties for database searches. The study utilized spectral libraries from NIST, MassIVE, and DeepNovo, with data split into 70% for training, 20% for validation, and 10% for testing. A total of 2.7 million spectra were analyzed, including 1.7 million spectra with a 2+ charge and 0.5 million spectra with a 3+ charge.


# Results
The proposed model significantly reduces the peptide search space by over 90%, enabling more efficient database searches. It achieves 8x–12x speedup and provides faster processing while maintaining comparable accuracy to traditional algorithms. Additionally, the model demonstrates up to 97% accuracy in predicting peptide properties, including length, missed cleavages, and modification status, directly from the spectra.


# Conclusions
Proteorift represents a major advancement in shotgun proteomics by leveraging deep learning to refine peptide search processes. By reducing the search space and improving processing speed without compromising accuracy, the model addresses key challenges in scalability and precision. The high accuracy in predicting peptide properties further highlights its potential for enhancing peptide deduction and database search workflows in proteomics research.