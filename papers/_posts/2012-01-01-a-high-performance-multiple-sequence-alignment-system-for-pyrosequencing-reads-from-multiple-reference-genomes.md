---
layout: paper
title: "A high performance multiple sequence alignment system for pyrosequencing reads from multiple reference genomes"
nickname: 2012-01-01-a-high-performance-multiple-sequence-alignment-system-for-pyrosequencing-reads-from-multiple-reference-genomes
authors: "Saeed, Fahad; Perez-Rathke, Alan; Gwarnicki, Jaroslaw; Berger-Wolf, Tanya; Khokhar, Ashfaq; "
year: "2012"
journal: "Academic Press Journal of parallel and distributed computing"
volume: 72
issue:
pages: 83-93
is_published: True
image: /assets/images/papers/elsevier.png
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
doi: "10.1016/j.jpdc.2011.08.001"
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

Genome resequencing with short reads generated from pyrosequencing generally relies on mapping the short reads against a single reference genome. However, mapping of reads from multiple reference genomes is not possible using a pairwise mapping algorithm. In order to align the reads w.r.t each other and the reference genomes, existing multiple sequence alignment(MSA) methods cannot be used because they do not take into account the position of these short reads with respect to the genome, and are highly inefficient for a large number of sequences. In this paper, we develop a highly scalable parallel algorithm based on domain decomposition, referred to as P-Pyro-Align, to align such a large number of reads from single or multiple reference genomes. The proposed alignment algorithm accurately aligns the erroneous reads, and has been implemented on a cluster of workstations using MPI library …
