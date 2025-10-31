---
layout: paper
title: "LBE: A Computational Load Balancing Algorithm for Speeding up Parallel Peptide Search in Mass-Spectrometry based Proteomics"
nickname: 2019-01-01-lbe--a-computational-load-balancing-algorithm-for-speeding-up-parallel-peptide-search-in-mass-spectrometry-based-proteomics
authors: "Haseeb, Muhammad; Afzali, Fatima; Saeed, Fahad; "
year: "2019"
journal: 
volume: 
issue:
pages: 
is_published: True
image: /assets/images/papers/ieee.png
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
doi: "10.1109/IPDPSW.2019.00040"
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

The most commonly employed method for peptide identification in mass-spectrometry based proteomics involves comparing experimentally obtained tandem MS/MS spectra against a set of theoretical MS/MS spectra. The theoretical MS/MS spectra data are predicted using protein sequence database. Most state-of-the-art peptide search algorithms index theoretical spectra data to quickly filter-in the relevant (similar) indexed spectra when searching an experimental MS/MS spectrum. Data filtration substantially reduces the required number of computationally expensive spectrum-to-spectrum comparison operations. However, the number of predicted (and indexed) theoretical spectra grows exponentially with increase in post-translational modifications creating a memory and I/O bottleneck. In this paper, we present a parallel algorithm, called LBE, for efficient partitioning of theoretical spectra data on a distributed …
