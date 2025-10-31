---
layout: paper
title: "A high performance algorithm for clustering of large-scale protein mass spectrometry data using multi-core architectures"
nickname: 2013-01-01-a-high-performance-algorithm-for-clustering-of-large-scale-protein-mass-spectrometry-data-using-multi-core-architectures
authors: "Saeed, Fahad; Hoffert, Jason D; Knepper, Mark A; "
year: "2013"
journal: 
volume: 
issue:
pages: 923-930
is_published: True
image: /assets/images/papers/springer.png
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
doi: "10.1145/2492517.2500245"
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

High-throughput mass spectrometers can produce thousands of peptide spectra from a single complex protein sample in a short amount of time. These data sets contain a substantial amount of redundancy (i.e. the same peptide is selected and identified multiple times in a single experiment) from peptides that may get selected multiple times in the liquid chromatography mass spectrometry (LC-MS/MS) experiment. The data from these mass spectrometers contain a substantial number of spectra that have low signal to noise (S/N) ratio and may not get interpreted due to poor quality. Recently, we presented a graph theoretic algorithm, CAMS (Clustering Algorithm for Mass Spectra) for clustering mass spectrometry data. CAMS utilized a novel metric, called a F-set, that allows accurate identification of the spectra that are similar with much higher accuracy and sensitivity than if single peak comparisons were performed …
