---
layout: paper
title: "Communication lower-bounds for distributed-memory computations for mass spectrometry based omics data"
nickname: 2022-01-01-communication-lower-bounds-for-distributed-memory-computations-for-mass-spectrometry-based-omics-data
authors: "Saeed, Fahad; Haseeb, Muhammad; Iyengar, SS; "
year: "2022"
journal: "Academic Press Journal of Parallel and Distributed Computing"
volume: 161
issue:
pages: 37-47
is_published: True
image: /assets/images/papers/elsevier.png
projects: [HPC-MS]
tags: []

# Text
fulltext:
pdf:
pdflink:
pmcid:
preprint: 
supplement:

# Links
doi: "10.1016/j.jpdc.2021.11.001"
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

Mass spectrometry (MS) based omics data analysis require significant time and resources. To date, few parallel algorithms have been proposed for deducing peptides from mass spectrometry-based data. However, these parallel algorithms were designed, and developed when the amount of data that needed to be processed was smaller in scale. In this paper, we prove that the communication bound that is reached by the existing parallel algorithms is Ω (m n+ 2 r q p), where m and n are the dimensions of the theoretical database matrix, q and r are dimensions of spectra, and p is the number of processors. We further prove that communication-optimal strategy with fast-memory M= m n+ 2 q r p can achieve Ω (2 m n q p) but is not achieved by any existing parallel proteomics algorithms till date. To validate our claim, we performed a meta-analysis of published parallel algorithms, and their performance results. We …
