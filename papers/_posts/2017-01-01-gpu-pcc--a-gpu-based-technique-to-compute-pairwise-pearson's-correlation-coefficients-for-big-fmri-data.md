---
layout: paper
title: "GPU-PCC: A GPU Based Technique to Compute Pairwise Pearson's Correlation Coefficients for Big fMRI Data"
nickname: 2017-01-01-gpu-pcc--a-gpu-based-technique-to-compute-pairwise-pearson's-correlation-coefficients-for-big-fmri-data
authors: "Eslami, Taban; Awan, Muaaz Gul; Saeed, Fahad; "
year: "2017"
journal: 
volume: 
issue:
pages: 723-728
is_published: True
image: /assets/images/papers/ieee.png
projects: []
tags: []

# Text
fulltext:
pdf:
pdflink:
pmcid:
preprint: 
supplement:

# Links
doi: "10.1145/3107411.3108173"
pmid:

# Data and code
github: []
neurovault:
openneuro: []
figshare:
figshare_names:
osf:
---
{% include JB/setup %}

# Abstract

Functional Magnetic Resonance Imaging (fMRI) is a non-invasive brain imaging technique for studying the brain's functional activities. Pearson's Correlation Coefficient is an important measure for capturing dynamic behaviors and functional connectivity between brain components. One bottleneck in computing Correlation Coefficients is the time it takes to process big fMRI data. In this paper, we propose GPU-PCC, a GPU based algorithm based on vector dot product, which is able to compute pairwise Pearson's Correlation Coefficients while performing computation once for each pair. Our method is able to compute Correlation Coefficients in an ordered fashion without the need to do post-processing reordering of coefficients. We evaluated GPU-PCC using synthetic and real fMRI data and compared it with sequential version of computing Correlation Coefficient on CPU and existing state-of-the-art GPU method. We …
