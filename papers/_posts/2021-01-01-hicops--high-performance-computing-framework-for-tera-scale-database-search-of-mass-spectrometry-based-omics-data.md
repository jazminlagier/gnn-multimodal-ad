---
layout: paper
title: "HiCOPS: High Performance Computing Framework for Tera-Scale Database Search of Mass Spectrometry based Omics Data"
nickname: 2021-01-01-hicops--high-performance-computing-framework-for-tera-scale-database-search-of-mass-spectrometry-based-omics-data
authors: "Haseeb, Muhammad; Saeed, Fahad; "
year: "2021"
journal: 
volume: 
issue:
pages: 
is_published: True
image: /assets/images/papers/biorxiv.png
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
doi: "10.48550/arXiv.2102.02286"
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

Database-search algorithms, that deduce peptides from Mass Spectrometry (MS) data, have tried to improve the computational efficiency to accomplish larger, and more complex systems biology studies. Existing serial, and high-performance computing (HPC) search engines, otherwise highly successful, are known to exhibit poor-scalability with increasing size of theoretical search-space needed for increased complexity of modern non-model, multi-species MS-based omics analysis. Consequently, the bottleneck for computational techniques is the communication costs of moving the data between hierarchy of memory, or processing units, and not the arithmetic operations. This post-Moore change in architecture, and demands of modern systems biology experiments have dampened the overall effectiveness of the existing HPC workflows. We present a novel efficient parallel computational method, and its implementation on memory-distributed architectures for peptide identification tool called HiCOPS, that enables more than 100-fold improvement in speed over most existing HPC proteome database search tools. HiCOPS empowers the supercomputing database search concept for comprehensive identification of peptides, and all their modified forms within a reasonable time-frame. We demonstrate this by searching Gigabytes of experimental MS data against Terabytes of databases where HiCOPS completes peptide identification in few minutes using 72 parallel nodes (1728 cores) compared to several weeks required by existing state-of-the-art tools using 1 node (24 cores); 100 minutes vs 5 weeks; 500x speedup. Finally, we formulate a …
