---
layout: paper
title: "TurboBC: A Memory Efficient and Scalable GPU Based Betweenness Centrality Algorithm in the Language of Linear Algebra"
nickname: 2021-01-01-turbobc--a-memory-efficient-and-scalable-gpu-based-betweenness-centrality-algorithm-in-the-language-of-linear-algebra
authors: "Artiles, Oswaldo; Saeed, Fahad; "
year: "2021"
journal: 
volume: 
issue:
pages: 1-10
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
doi: "10.1145/3458744.3474047"
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

Betweenness centrality (BC) is a shortest path centrality metric used to measure the influence of individual vertices or edges on huge graphs that are used for modeling and analysis of human brain, omics data, or social networks. The application of the BC algorithm to modern graphs must deal with the size of the graphs, as well with highly irregular data-access patterns. These challenges are particularly important when the BC algorithm is implemented on Graphics Processing Units (GPU), due to the limited global memory of these processors, as well as the decrease in performance due to the load unbalance resulting from processing irregular data structures. In this paper, we present the first GPU based linear-algebraic formulation and implementation of BC, called TurboBC, a set of memory efficient BC algorithms that exhibits good performance and high scalability on unweighted, undirected or directed sparse …
