---
layout: paper
title: "A Parallel Peptide Indexer and Decoy Generator for Crux Tide using OpenMP"
nickname: 2016-01-01-a-parallel-peptide-indexer-and-decoy-generator-for-crux-tide-using-openmp
authors: "Maabreh, Majdi; Gupta, Ajay; Saeed, Fahad; "
year: "2016"
journal: "IEEE 2016 International Conference on High Performance Computing Simulation (HPCS)"
volume: 
issue:
pages: 411-418
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
doi: "10.1109/HPCSim.2016.7568364"
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

Target-Decoy database is a common dependable strategy used in Peptide-Spectrum-Matching (PSM) for quality assessment. In this, a set of decoy labeled peptides are injected to the database and indexed along with real peptides. Crux Tide is a fast search engine that supports indexing peptides' database and decoys generation. In Tide, indexing FASTA files and generating decoys is a computationally expensive process. In this paper we first analyze the serial Tide indexer and decoy generator algorithm and, then describe a parallel shared memory solution. Our proposed technique utilizes a clever hashing technique to localize the process, and breaks up the processing dependency among threads. The developed parallel versions are able to reduce the computational complexity by approximately 50% and 25% of the sequential time using 4 and 8 threads, respectively. Moreover, our proposed solution could …
