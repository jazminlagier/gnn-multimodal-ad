---
layout: paper
title: "Parallel Algorithm for Center Star Sequence and Alignments with Applications to Short Reads"
nickname: 2010-01-01-parallel-algorithm-for-center-star-sequence-and-alignments-with-applications-to-short-reads
authors: "Saeed, Fahad; Khokhar, Ashfaq; "
year: "2010"
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
doi: ""
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

With the advent of fast sequencing techniques, such as 454 and Solexa, the number of sequences that need to be aligned and processed are reaching up to one billion sequences [1]. The alignment of these sequences with the reference genome is one of the basic steps in mapping, alignments, and sequence analysis related problems. Aligning of such a large number of sequences using traditional multiple sequence alignment algorithms is computationally infeasible. In this paper we present a highly scalable parallel algorithm to find the center star sequence and perform approximate alignments of such a large number of sequences. The proposed algorithm has been implemented on a cluster of workstations using MPI library, and experimental results to find the Center Sequence for up to 6.4 million sequences are presented. These results include detailed computation and communication complexity analysis as …
