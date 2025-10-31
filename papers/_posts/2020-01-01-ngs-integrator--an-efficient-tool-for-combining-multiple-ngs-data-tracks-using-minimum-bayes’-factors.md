---
layout: paper
title: "NGS-Integrator: An efficient tool for combining multiple NGS data tracks using minimum Bayes’ factors"
nickname: 2020-01-01-ngs-integrator--an-efficient-tool-for-combining-multiple-ngs-data-tracks-using-minimum-bayes’-factors
authors: "Wen, Bronte; Jung, Hyun Jun; Chen, Lihe; Saeed, Fahad; Knepper, Mark A; "
year: "2020"
journal: "BioMed Central BMC genomics"
volume: 21
issue:
pages: 1-7
is_published: True
image: /assets/images/papers/oxford.png
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
doi: "10.1186/s12864-020-07220-7"
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

Background Next-generation sequencing (NGS) is widely used for genome-wide identification and quantification of DNA elements involved in the regulation of gene transcription. Studies that generate multiple high-throughput NGS datasets require data integration methods for two general tasks: 1) generation of genome-wide data tracks representing an aggregate of multiple replicates of the same experiment; and 2) combination of tracks from different experimental types that provide complementary information regarding the location of genomic features such as enhancers.   Results NGS-Integrator is a Java-based command line application, facilitating efficient integration of multiple genome-wide NGS datasets. NGS-Integrator first transforms all input data tracks using the complement of the minimum Bayes’ factor so that all values are expressed in the range [0,1] representing the probability of a true signal given the …
