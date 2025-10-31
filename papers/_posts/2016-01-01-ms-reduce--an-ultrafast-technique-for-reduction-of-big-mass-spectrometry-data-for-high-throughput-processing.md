---
layout: paper
title: "MS-REDUCE: an ultrafast technique for reduction of big mass spectrometry data for high-throughput processing"
nickname: 2016-01-01-ms-reduce--an-ultrafast-technique-for-reduction-of-big-mass-spectrometry-data-for-high-throughput-processing
authors: "Awan, Muaaz Gul; Saeed, Fahad; "
year: "2016"
journal: "Oxford University Press Bioinformatics"
volume: 32
issue:
pages: 1518-1526
is_published: True
image: /assets/images/papers/oxford.png
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
doi: "10.1093/bioinformatics/btw023"
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

Motivation: Modern proteomics studies utilize high-throughput mass spectrometers which can produce data at an astonishing rate. These big mass spectrometry (MS) datasets can easily reach peta-scale level creating storage and analytic problems for large-scale systems biology studies. Each spectrum consists of thousands of peaks which have to be processed to deduce the peptide. However, only a small percentage of peaks in a spectrum are useful for peptide deduction as most of the peaks are either noise or not useful for a given spectrum. This redundant processing of non-useful peaks is a bottleneck for streaming high-throughput processing of big MS data. One way to reduce the amount of computation required in a high-throughput environment is to eliminate non-useful peaks. Existing noise removing algorithms are limited in their data-reduction capability and are compute intensive making them …
