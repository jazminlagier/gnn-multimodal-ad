---
layout: paper
title: "Scalable data structure to compress next-generation sequencing files and its application to compressive genomics"
nickname: 2017-01-01-scalable-data-structure-to-compress-next-generation-sequencing-files-and-its-application-to-compressive-genomics
authors: "Pérez, Sandino Vargas; Saeed, Fahad; "
year: "2017"
journal: "IEEE 2017 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)"
volume: 
issue:
pages: 1923-1928
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
doi: "10.1109/BIBM.2017.8217953"
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

It is now possible to compress and decompress large-scale Next-Generation Sequencing files taking advantage of high-performance computing techniques. To this end, we have recently introduced a scalable hybrid parallel algorithm, called phyNGSC, which allows fast compression as well as decompression of big FASTQ datasets using distributed and shared memory programming models via MPI and OpenMP. In this paper we present the design and implementation of a novel parallel data structure which lessens the dependency on decompression and facilitates the handling of DNA sequences in their compressed state using fine-grained decompression in a technique that is identified as in compresso data processing. Using our data structure compression and decompression throughputs of up to 8.71 GB/s and 10.12 GB/s were observed. Our proposed structure and methodology brings us one step closer to …
