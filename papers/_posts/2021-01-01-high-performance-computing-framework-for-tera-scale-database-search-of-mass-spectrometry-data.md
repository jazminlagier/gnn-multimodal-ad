---
layout: paper
title: "High performance computing framework for tera-scale database search of mass spectrometry data"
nickname: 2021-01-01-high-performance-computing-framework-for-tera-scale-database-search-of-mass-spectrometry-data
authors: "Haseeb, Muhammad; Saeed, Fahad; "
year: "2021"
journal: "Nature Publishing Group Nature computational science"
volume: 1
issue:
pages: 550-561
is_published: True
image: /assets/images/papers/nature.png
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
doi: "10.1038/s43588-021-00113-z"
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

Database peptide search algorithms deduce peptides from mass spectrometry data. There has been substantial effort in improving their computational efficiency to achieve larger and more complex systems biology studies. However, modern serial and high-performance computing (HPC) algorithms exhibit suboptimal performance mainly due to their ineffective parallel designs (low resource utilization) and high overhead costs. We present an HPC framework, called HiCOPS, for efficient acceleration of the database peptide search algorithms on distributed-memory supercomputers. HiCOPS provides, on average, more than tenfold improvement in speed and superior parallel performance over several existing HPC database search software. We also formulate a mathematical model for performance analysis and optimization, and report near-optimal results for several key metrics including strong-scale efficiency …
