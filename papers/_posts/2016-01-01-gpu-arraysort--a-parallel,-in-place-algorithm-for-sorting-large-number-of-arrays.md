---
layout: paper
title: "GPU-ArraySort: A parallel, in-place algorithm for sorting large number of arrays"
nickname: 2016-01-01-gpu-arraysort--a-parallel,-in-place-algorithm-for-sorting-large-number-of-arrays
authors: "Awan, Muaaz; Saeed, Fahad; "
year: "2016"
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
doi: "10.1109/ICPPW.2016.27"
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

Modern day analytics deals with big datasets from diverse fields. For many application the data is in the form of an array which consists of large number of smaller arrays. Existing techniques focus on sorting a single large array and cannot be used for sorting large number of smaller arrays in an efficient manner. Currently no such algorithm is available which can sort such large number of arrays utilizing the massively parallel architecture of GPU devices. In this paper we present a highly scalable parallel algorithm, called GPU-ArraySort, for sorting large number of arrays using a GPU. Our algorithm performs in-place operations and makes minimum use of any temporary run-time memory. Our results indicate that we can sort up to 2 million arrays having 1000 elements each, within few seconds. We compare our results with the unorthodox tagged array sorting technique based on NVIDIAs Thrust library. GPU-ArraySort …
