---
layout: project
title: ProteoRift
contributors: [Prof-S, usman]
handle: ML-MS-ProteoRift
status: analysis
type: software

# Optional
website: 
grant:
grant_url:
image: /assets/images/projects/ML-MS.png
tagline: End-to-end ML pipeline for Data Dependent Acquisition (DDA) MS data 
tags: [software]

# Data and code
github: ["https://github.com/pcdslab/ProteoRift", "https://github.com/pcdslab/SpeCollate"]
neurovault:
openneuro:
figshare:
figshare_names:
osf:
---
{% include JB/setup %}

This project will design and develop machine-learning software infrastructure that can work in an end-to-end fashion on mass spectrometry (MS) data acquired through data-dependent acquisition. The proposed ML software infrastructure will enable identification of novel peptides/proteins, and insights into microbiome communities and their effects on human health, agriculture, and environments. 

Database search algorithms reduce the number of potential candidate peptides against which scoring needs to be performed using a single (i.e. mass) property for filtering. While useful, filtering based on one property may lead to exclusion of non-abundant spectra and uncharacterized peptides -potentially exacerbating the streetlight effect. We are designing, developing and refining ProteoRift, a novel attention and multitask deep-network, which can predict multiple peptide properties (length, missed cleavages, and modification status) directly from spectra. We demonstrate that ProteoRift can predict these properties with up to 97% accuracy resulting in search-space reduction by more than 90%. As a result, our end-to-end pipeline, integrated with SpeCollate, is shown to exhibit 8x to 12x speedups with peptide deduction accuracy comparable to algorithmic techniques. 

The development of such end-to-end pipelines will enable proteomics scientists to use ML infrastructure in an end-to-end fashion. 


## Status 

The method development for this work is on going