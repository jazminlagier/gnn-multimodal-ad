---
layout: paper
title: "GPU-DFC: A GPU-based parallel algorithm for computing dynamic-functional connectivity of big fMRI data"
nickname: 2019-01-01-gpu-dfc--a-gpu-based-parallel-algorithm-for-computing-dynamic-functional-connectivity-of-big-fmri-data
authors: "Eslami, Taban; Saeed, Fahad; "
year: "2019"
journal: 
volume: 
issue:
pages: 
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
doi: "10.1109/BigDataService.2019.00022"
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

Studying dynamic-functional connectivity (DFC) using fMRI data of the brain gives much richer information to neuroscientists than studying the brain as a static entity. Mining of dynamic connectivity graphs from these brain studies can be used to classify diseased versus healthy brains. However, constructing and mining dynamic-functional connectivity graphs of the brain can be time consuming due to size of fMRI data. In this paper, we propose a highly scalable GPU-based parallel algorithm called GPU-DFC for computing dynamic-functional connectivity of fMRI data both at region and voxel level. Our algorithm exploits sparsification of correlation matrix and stores them in CSR format. Further reduction in the correlation matrix is achieved by parallel decomposition techniques. Our GPU-DFC algorithm achieves 2 times speed-up for computing dynamic correlations compared to state-of-the-art GPU-based techniques …
