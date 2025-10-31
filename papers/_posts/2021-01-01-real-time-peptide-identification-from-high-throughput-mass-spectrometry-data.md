---
layout: paper
title: "Real-time peptide identification from high-throughput mass-spectrometry data"
nickname: 2021-01-01-real-time-peptide-identification-from-high-throughput-mass-spectrometry-data
authors: "Kumar, Sumesh; Saeed, Fahad; "
year: "2021"
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
doi: "10.1145/3459930.3470856"
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

Peptide deduction remains one of the most challenging research problems in the large-scale study of proteomes using high-throughput Mass Spectrometers. The identification of large number of proteins from complex biological samples can be carried out in two steps: 1) tryptic digestion of protein sample to isolate constituent peptides, and then generating MS/MS data using high-thought put mass spectrometers; 2) Once the data is generated various method such as database-search tools are used to compare mass-spectrometry data against a repository of known peptides. Advances in the MS instrumentation now allow generation of high-resolution data in massive volume and velocity making traditional MS based algorithms a bottleneck in the overall workflows [4]. New generation of state-of-the-art database search tools are now capable of producing highquality matches with impressively low FDR; however, the search time usually takes somewhere between a few weeks to a few months depending on the size of database and search parameters. To accelerate the overall search times, several studies have been proposed which target this computational bottleneck by exploiting specialized hardware architectures including HPC compute clusters and GPUs [2],[1]. Even with these accelerated pipelines the dream of realizing a true real-time processing and deduction of peptides from MS data is a far from realization. One bottleneck preventing the design of true real-time processing of MS based data is the cost of communication of the data required for the existing workflows [3] ie moving the data from storage to computational nodes and across …
