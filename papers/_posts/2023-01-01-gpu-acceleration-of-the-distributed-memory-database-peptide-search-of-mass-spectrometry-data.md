---
layout: paper
title: "GPU-acceleration of the distributed-memory database peptide search of mass spectrometry data"
nickname: 2023-01-01-gpu-acceleration-of-the-distributed-memory-database-peptide-search-of-mass-spectrometry-data
authors: "Haseeb, Muhammad; Saeed, Fahad; "
year: "2023"
journal: "Nature Publishing Group UK London Scientific Reports"
volume: 13
issue:
pages: 18713
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
doi: "10.1038/s41598-023-43033-w"
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

Database peptide search is the primary computational technique for identifying peptides from the mass spectrometry (MS) data. Graphical Processing Units (GPU) computing is now ubiquitous in the current-generation of high-performance computing (HPC) systems, yet its application in the database peptide search domain remains limited. Part of the reason is the use of sub-optimal algorithms in the existing GPU-accelerated methods resulting in significantly inefficient hardware utilization. In this paper, we design and implement a new-age CPU-GPU HPC framework, called GiCOPS, for efficient and complete GPU-acceleration of the modern database peptide search algorithms on supercomputers. Our experimentation shows that the GiCOPS exhibits between 1.2 to 5 speed improvement over its CPU-only predecessor, HiCOPS, and over 10 improvement over several existing GPU-based database search …
