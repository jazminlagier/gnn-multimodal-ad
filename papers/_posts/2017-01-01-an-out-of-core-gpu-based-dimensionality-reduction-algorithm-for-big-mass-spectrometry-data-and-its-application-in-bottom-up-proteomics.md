---
layout: paper
title: "An out-of-core gpu based dimensionality reduction algorithm for big mass spectrometry data and its application in bottom-up proteomics"
nickname: 2017-01-01-an-out-of-core-gpu-based-dimensionality-reduction-algorithm-for-big-mass-spectrometry-data-and-its-application-in-bottom-up-proteomics
authors: "Awan, Muaaz Gul; Saeed, Fahad; "
year: "2017"
journal: 
volume: 
issue:
pages: 550-555
is_published: True
image: /assets/images/papers/acm.png
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
doi: "10.1145/3107411.3107466"
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

Modern high resolution Mass Spectrometry instruments can generate millions of spectra in a single systems biology experiment. Each spectrum consists of thousands of peaks but only a small number of peaks actively contribute to deduction of peptides. Therefore, pre-processing of MS data to detect noisy and non-useful peaks are an active area of research. Most of the sequential noise reducing algorithms are impractical to use as a pre-processing step due to high time-complexity. In this paper, we present a GPU based dimensionality-reduction algorithm, called G-MSR, for MS2 spectra. Our proposed algorithm uses novel data structures which optimize the memory and computational operations inside GPU. These novel data structures include Binary Spectra and Quantized Indexed Spectra (QIS). The former helps in communicating essential information between CPU and GPU using minimum amount of data …
