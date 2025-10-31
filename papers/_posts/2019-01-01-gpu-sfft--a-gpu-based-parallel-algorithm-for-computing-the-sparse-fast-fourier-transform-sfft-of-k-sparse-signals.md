---
layout: paper
title: "GPU-SFFT: A GPU based parallel algorithm for computing the Sparse Fast Fourier Transform (SFFT) of k-sparse signals"
nickname: 2019-01-01-gpu-sfft--a-gpu-based-parallel-algorithm-for-computing-the-sparse-fast-fourier-transform-sfft-of-k-sparse-signals
authors: "Artiles, Oswaldo; Saeed, Fahad; "
year: "2019"
journal: "IEEE 2019 IEEE International Conference on Big Data (Big Data)"
volume: 
issue:
pages: 3303-3311
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
doi: "10.1109/BigData47090.2019.9006579"
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

The Sparse Fast Fourier Transform (MIT-SFFT) is an algorithm to compute the discrete Fourier transform of a signal with a sublinear time complexity, i.e. algorithms with runtime complexity proportional to the sparsity level k, where k is the number of non-zero coefficients of the signal in the frequency domain. In this paper, we propose a highly scalable GPU-based parallel algorithm called GPU-SFFT for computing the SFFT of k-sparse signals. Our implementation of GPU-SFFT is based on parallel optimizations that leads to enormous speedups. These include carefully crafting parallel regions in the sequential MIT-SFFT code to exploit parallelism, and minimizing data movement between the CPU and the GPU. This allows us to exploit extreme parallelism for the CPU-GPU architectures and to maximize the number of concurrent threads executing instructions. Our experiments show that our designed CPU-GPU …
