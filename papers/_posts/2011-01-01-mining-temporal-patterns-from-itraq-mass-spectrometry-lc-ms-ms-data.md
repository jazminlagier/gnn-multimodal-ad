---
layout: paper
title: "Mining temporal patterns from iTRAQ mass spectrometry (LC-MS/MS) data"
nickname: 2011-01-01-mining-temporal-patterns-from-itraq-mass-spectrometry-lc-ms-ms-data
authors: "Saeed, Fahad; Pisitkun, Trairak; Knepper, Mark A; Hoffert, Jason D; "
year: "2011"
journal: 
volume: 
issue:
pages: 
is_published: True
image: /assets/images/papers/biorxiv.png
projects: [ML-MS]
tags: []

# Text
fulltext:
pdf:
pdflink:
pmcid:
preprint: 
supplement:

# Links
doi: "10.48550/arXiv.1104.5510"
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

Large-scale proteomic analysis is emerging as a powerful technique in biology and relies heavily on data acquired by state-of-the-art mass spectrometers. As with any other field in Systems Biology, computational tools are required to deal with this ocean of data. iTRAQ (isobaric Tags for Relative and Absolute quantification) is a technique that allows simultaneous quantification of proteins from multiple samples. Although iTRAQ data gives useful insights to the biologist, it is more complex to perform analysis and draw biological conclusions because of its multi-plexed design. One such problem is to find proteins that behave in a similar way (i.e. change in abundance) among various time points since the temporal variations in the proteomics data reveal important biological information. Distance based methods such as Euclidian distance or Pearson coefficient, and clustering techniques such as k-mean etc, are not able to take into account the temporal information of the series. In this paper, we present an linear-time algorithm for clustering similar patterns among various iTRAQ time course data irrespective of their absolute values. The algorithm, referred to as Temporal Pattern Mining(TPM), maps the data from a Cartesian plane to a discrete binary plane. After the mapping a dynamic programming technique allows mining of similar data elements that are temporally closer to each other. The proposed algorithm accurately clusters iTRAQ data that are temporally closer to each other with more than 99% accuracy. Experimental results for different problem sizes are analyzed in terms of quality of clusters, execution time and scalability for large data sets …
