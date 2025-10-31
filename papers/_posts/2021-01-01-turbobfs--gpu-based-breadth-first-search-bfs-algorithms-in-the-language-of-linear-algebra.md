---
layout: paper
title: "TurboBFS: GPU Based Breadth-First Search (BFS) Algorithms in the Language of Linear Algebra"
nickname: 2021-01-01-turbobfs--gpu-based-breadth-first-search-bfs-algorithms-in-the-language-of-linear-algebra
authors: "Artiles, Oswaldo; Saeed, Fahad; "
year: "2021"
journal: "IEEE 2021 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)"
volume: 
issue:
pages: 520-528
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
doi: "10.1109/IPDPSW52791.2021.00084"
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

Graphs that are used for modeling of human brain, omics data, or social networks are huge, and manual inspection of these graph is impossible. A popular, and fundamental, method used for making sense of these large graphs is the well-known Breadth-First Search (BFS) algorithm. However, BFS suffers from large computational cost especially for big graphs of interest. More recently, the use of Graphics processing units (GPU) has been promising, but challenging because of limited global memory of GPU’s, and irregular structures of real-world graphs. In this paper, we present a GPU based linear-algebraic formulation and implementation of BFS, called TurboBFS, that exhibits excellent scalability on unweighted, undirected or directed sparse graphs of arbitrary structure. We demonstrate that our algorithms obtain up to 40 GTEPs, and are on average 15.7x, 5.8x, and 1.8x faster than the other state-of-the-art …
