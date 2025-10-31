---
layout: paper
title: "A Hybrid MPI-OpenMP Strategy to Speedup the Compression of Big Next-Generation Sequencing Datasets"
nickname: 2017-01-01-a-hybrid-mpi-openmp-strategy-to-speedup-the-compression-of-big-next-generation-sequencing-datasets
authors: "Vargas-Perez, Sandino; Saeed, Fahad; "
year: "2017"
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
doi: "10.1109/TPDS.2017.2692782"
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

DNA sequencing has moved into the realm of Big Data due to the rapid development of high-throughput, low cost Next-Generation Sequencing (NGS) technologies. Sequential data compression solutions that once were sufficient to efficiently store and distribute this information are now falling behind. In this paper we introduce phyNGSC, a hybrid MPI-OpenMP strategy to speedup the compression of big NGS data by combining the features of both distributed and shared memory architectures. Our algorithm balances work-load among processes and threads, alleviates memory latency by exploiting locality, and accelerates I/O by reducing excessive read/write operations and inter-node message exchange. To make the algorithm scalable, we introduce a novel timestamp-based file structure that allows us to write the compressed data in a distributed and non-deterministic fashion while retaining the capability of …
