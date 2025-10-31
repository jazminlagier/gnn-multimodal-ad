---
layout: paper
title: "Parallel sampling-pipeline for indefinite stream of heterogeneous graphs using OpenCL for FPGAs"
nickname: 2018-01-01-parallel-sampling-pipeline-for-indefinite-stream-of-heterogeneous-graphs-using-opencl-for-fpgas
authors: "Tariq, Muhammad Usman; Saeed, Fahad; "
year: "2018"
journal: "IEEE 2018 IEEE International Conference on Big Data (Big Data)"
volume: 
issue:
pages: 4752-4761
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
doi: "10.1109/BigData.2018.8621979"
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

In the field of data science, a huge amount of data, generally represented as graphs, needs to be processed and analyzed. It is of utmost importance that this data be processed swiftly and efficiently to save time and energy. The volume and velocity of data, along with irregular access patterns in graph data structures, pose challenges in terms of analysis and processing. Further, a big chunk of time and energy is spent on analyzing these graphs on large compute clusters and/or data-centers. Filtering and refining of data using graph sampling techniques are one of the most effective ways to speed up the analysis. Efficient accelerators, such as FPGAs, have proven to significantly lower the energy cost of running an algorithm. To this end, we present the design and implementation of a parallel graph sampling technique, for a large number of input graphs streaming into a FPGA. A parallel approach using OpenCL for …
