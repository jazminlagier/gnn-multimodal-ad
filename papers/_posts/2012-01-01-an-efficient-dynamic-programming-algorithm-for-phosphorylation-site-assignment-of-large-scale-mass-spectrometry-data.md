---
layout: paper
title: "An efficient dynamic programming algorithm for phosphorylation site assignment of large-scale mass spectrometry data"
nickname: 2012-01-01-an-efficient-dynamic-programming-algorithm-for-phosphorylation-site-assignment-of-large-scale-mass-spectrometry-data
authors: "Saeed, Fahad; Pisitkun, Trairak; Hoffert, Jason D; Wang, Guanghui; Gucek, Marjan; Knepper, Mark A; "
year: "2012"
journal: "IEEE 2012 IEEE International Conference on Bioinformatics and Biomedicine Workshops"
volume: 
issue:
pages: 618-625
is_published: True
image: /assets/images/papers/ieee.png
projects: [ML-MS]
tags: []

# Text
fulltext:
pdf:
pdflink:
pmcid:
preprint: 
supplement:

# Links
doi: "10.1109/BIBMW.2012.6470210"
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

Phosphorylation site assignment of large-scale data from high throughput tandem mass spectrometry (LC-MS/MS) data is an important aspect of phosphoproteomics. Correct assignment of phosphorylated residue(s) is important for functional interpretation of the data within a biological context. Common search algorithms (Sequest etc.) for mass spectrometry data are not designed for accurate site assignment; thus, additional algorithms are needed. In this paper, we propose a linear-time and linear-space dynamic programming strategy for phosphorylation site assignment. The algorithm, referred to as PhosSA, optimizes the objective function defined as the summation of peak intensities that are associated with theoretical phosphopeptide fragmentation ions. Quality control is achieved through the use of a post-processing criteria whose value is indicative of the signal-to-noise (S/N) properties and redundancy of the …
