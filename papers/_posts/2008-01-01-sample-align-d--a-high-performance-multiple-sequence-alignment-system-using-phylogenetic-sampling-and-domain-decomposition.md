---
layout: paper
title: "Sample-align-d: A high performance multiple sequence alignment system using phylogenetic sampling and domain decomposition"
nickname: 2008-01-01-sample-align-d--a-high-performance-multiple-sequence-alignment-system-using-phylogenetic-sampling-and-domain-decomposition
authors: "Saeed, Fahad; Khokhar, Ashfaq; "
year: "2008"
journal: "IEEE 2008 IEEE International Symposium on Parallel and Distributed Processing"
volume: 
issue:
pages: 1-9
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
doi: "10.1109/IPDPS.2008.4536174"
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

Multiple sequence alignment (MSA) is one of the most computationally intensive tasks in Computational Biology. Existing best known solutions for multiple sequence alignment take several hours (in some cases days) of computation time to align, for example, 2000 homologous sequences of average length 300. Inspired by the Sample Sort approach in parallel processing, in this paper we propose a highly scalable multiprocessor solution for the MSA problem in phylogenetically diverse sequences. Our method employs an intelligent scheme to partition the set of sequences into smaller subsets using k- mer count based similarity index, referred to as k-mer rank. Each subset is then independently aligned in parallel using any sequential approach. Further fine tuning of the local alignments is achieved using constraints derived from a global ancestor of the entire set. The proposed sample-align-D algorithm has been …
