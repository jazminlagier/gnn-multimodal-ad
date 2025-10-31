---
layout: paper
title: "PhosSA: Fast and accurate phosphorylation site assignment algorithm for mass spectrometry data"
nickname: 2013-01-01-phossa--fast-and-accurate-phosphorylation-site-assignment-algorithm-for-mass-spectrometry-data
authors: "Saeed, Fahad; Pisitkun, Trairak; Hoffert, Jason D; Rashidian, Sara; Wang, Guanghui; Gucek, Marjan; Knepper, Mark A; "
year: "2013"
journal: "BioMed Central Proteome science"
volume: 11
issue:
pages: 1-15
is_published: True
image: /assets/images/papers/oxford.png
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
doi: "10.1186/1477-5956-11-S1-S14"
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

Phosphorylation site assignment of high throughput tandem mass spectrometry (LC-MS/MS) data is one of the most common and critical aspects of phosphoproteomics. Correctly assigning phosphorylated residues helps us understand their biological significance. The design of common search algorithms (such as Sequest, Mascot etc.) do not incorporate site assignment; therefore additional algorithms are essential to assign phosphorylation sites for mass spectrometry data. The main contribution of this study is the design and implementation of a linear time and space dynamic programming strategy for phosphorylation site assignment referred to as PhosSA. The proposed algorithm uses summation of peak intensities associated with theoretical spectra as an objective function. Quality control of the assigned sites is achieved using a post-processing redundancy criteria that indicates the signal-to-noise ratio …
