---
layout: paper
title: "Predicting peptide properties from mass spectrometry data using deep attention-based multitask network and uncertainty quantification"
nickname: ProteoRift-paper
authors: "Tariq, Usman; Saeed, Fahad; "
year: "2024"
journal: "bioRxiv"
volume: 
issue:
pages: 1-22
is_published: False
image: /assets/images/papers/biorxiv.png
projects: [ML-MS]
tags: [preprint]

# Text
fulltext: https://www.biorxiv.org/content/10.1101/2024.08.21.609035v1
pdf:
pdflink: https://www.biorxiv.org/content/10.1101/2024.08.21.609035v1.full.pdf
pmcid:
preprint: 
supplement:

# Links
doi: "10.1101/2024.08.21.609035"
pmid:

# Data and code
github: [https://github.com/pcdslab/ProteoRift]
neurovault:
openneuro: []
figshare:
figshare_names:
osf:
---
{% include JB/setup %}

# Abstract

Database search algorithms reduce the number of potential candidate peptides against which scoring needs to be performed using a single (i.e. mass) property for filtering. While useful, filtering based on one property may lead to exclusion of non-abundant spectra and uncharacterized peptides -potentially exacerbating the streetlight effect. Here we present ProteoRift, a novel attention and multitask deep-network, which can predict multiple peptide properties (length, missed cleavages, and modification status) directly from spectra. We demonstrate that ProteoRift can predict these properties with up to 97% accuracy resulting in search-space reduction by more than 90%. As a result, our end-to-end pipeline is shown to exhibit 8x to 12x speedups with peptide deduction accuracy comparable to algorithmic techniques. We also formulate two uncertainty estimation metrics, which can distinguish between in-distribution and out-of-distribution data (ROC AUC 0.99) and predict high-scoring mass spectra against correct peptide (ROC AUC 0.94). These models and metrics are integrated in an end-to-end ML pipeline available at https://github.com/pcdslab/ProteoRift.
