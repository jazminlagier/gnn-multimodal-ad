---
layout: paper
title: "A parallel algorithm for compression of big next-generation sequencing datasets"
nickname: 2015-01-01-a-parallel-algorithm-for-compression-of-big-next-generation-sequencing-datasets
authors: "Pérez, Sandino Vargas; Saeed, Fahad; "
year: "2015"
journal: "IEEE 2015 IEEE Trustcom/BigDataSE/ISPA"
volume: 3
issue:
pages: 196-201
is_published: True
image: /assets/images/papers/ieee.png
projects: []
tags: []

# Text
fulltext:
pdf:
pdflink:
pmcid:
preprint: 
supplement:

# Links
doi: "10.1109/Trustcom.2015.632"
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

The amount of big data from high-throughput Next-Generation Sequencing (NGS) techniques represents various challenges such as storage, analysis and transmission of massive datasets. One solution to storage and transmission of data is compression using specialized compression algorithms. The existing specialized algorithms suffer from poor scalability with increasing size of the datasets and best available solutions can take hours to compress gigabytes of data. Compression and decompression using these techniques for peta-scale data sets is prohibitively expensive in terms of time and energy. In this paper we introduce paraDSRC, a parallel implementation of the DNA Sequence Reads Compression (DSRC) application using a message passing model that presents reduction of the compression time complexity by a factor of O(1/p) (where p is the number of processing units). Our experimental results …
