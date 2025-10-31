---
layout: paper
title: "Efficient shared peak counting in database peptide search using compact data structure for fragment-ion index"
nickname: 2019-01-01-efficient-shared-peak-counting-in-database-peptide-search-using-compact-data-structure-for-fragment-ion-index
authors: "Haseeb, Muhammad; Saeed, Fahad; "
year: "2019"
journal: "IEEE 2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)"
volume: 
issue:
pages: 275-278
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
doi: "10.1109/BIBM47256.2019.8983152"
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

Database search is the most commonly employed method for identification of peptides from MS/MS spectra data. The search involves comparing experimentally obtained MS/MS spectra against a set of theoretical spectra predicted from a protein sequence database. One of the most commonly employed similarity metrics for spectral comparison is the shared-peak count between a pair of MS/MS spectra. Most modern methods index all generated fragment-ion data from theoretical spectra to speed up the shared peak count computations between a given experimental spectrum and all theoretical spectra. However, the bottleneck for this method is the gigantic memory footprint of fragment-ion index that leads to non-scalable solutions. In this paper, we present a novel data structure, called Compact Fragment-Ion Index Representation (CFIR), that efficiently compresses highly redundant ion-mass information in the data …
