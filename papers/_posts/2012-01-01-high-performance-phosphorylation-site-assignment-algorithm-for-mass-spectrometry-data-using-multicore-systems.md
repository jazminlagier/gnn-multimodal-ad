---
layout: paper
title: "High performance phosphorylation site assignment algorithm for mass spectrometry data using multicore systems"
nickname: 2012-01-01-high-performance-phosphorylation-site-assignment-algorithm-for-mass-spectrometry-data-using-multicore-systems
authors: "Saeed, Fahad; Hoffert, Jason; Pisitkun, Trairak; Knepper, Mark; "
year: "2012"
journal: 
volume: 
issue:
pages: 667-672
is_published: True
image: /assets/images/papers/acm.png
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
doi: "10.1145/2382936.2383056"
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

Phosphorylation site assignment of high throughput tandem mass spectrometry (LC-MS/MS) data is one of the most common and critical aspects of phosphoproteomics. Although a number of tools have been proposed for automated site assignments, most of them have been limited in scalability or quality. In this paper, we propose a parallelized version of the PhosSA algorithm (called P-PhosSA) that we recently introduced for site assignment. A domain decomposition strategy is introduced for site assignment and the decomposed data is executed on multiple cores. The algorithm has been parallelized using Java Threads executing on multicore systems. The parallelized algorithm is tested using experimentally generated data sets of peptides with known phosphorylation sites while varying the fragmentation strategy (CID, HCD) and molarities of the peptides. The algorithm is also compatible with various peptide …
