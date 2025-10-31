---
layout: paper
title: "ASD-SAENet: a sparse autoencoder, and deep-neural network model for detecting autism spectrum disorder (ASD) using fMRI data"
nickname: 2021-01-01-asd-saenet--a-sparse-autoencoder,-and-deep-neural-network-model-for-detecting-autism-spectrum-disorder-asd-using-fmri-data
authors: "Almuqhim, Fahad; Saeed, Fahad; "
year: "2021"
journal: "Frontiers Media SA Frontiers in Computational Neuroscience"
volume: 15
issue:
pages: 654315
is_published: True
image: /assets/images/papers/frontiers.png
projects: [ML-brain-imaging]
tags: []

# Text
fulltext:
pdf:
pdflink:
pmcid:
preprint: 
supplement:

# Links
doi: "10.3389/fncom.2021.654315/full"
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

Autism spectrum disorder (ASD) is a heterogenous neurodevelopmental disorder which is characterized by impaired communication, and limited social interactions. The shortcomings of current clinical approaches which are based exclusively on behavioral observation of symptomology, and poor understanding of the neurological mechanisms underlying ASD necessitates the identification of new biomarkers that can aid in study of brain development, and functioning, and can lead to accurate and early detection of ASD. In this paper, we developed a deep-learning model called ASD-SAENet for classifying patients with ASD from typical control subjects using fMRI data. We designed and implemented a sparse autoencoder (SAE) which results in optimized extraction of features that can be used for classification. These features are then fed into a deep neural network (DNN) which results in superior classification of fMRI brain scans more prone to ASD. Our proposed model is trained to optimize the classifier while improving extracted features based on both reconstructed data error and the classifier error. We evaluated our proposed deep-learning model using publicly available Autism Brain Imaging Data Exchange (ABIDE) dataset collected from 17 different research centers, and include more than 1,035 subjects. Our extensive experimentation demonstrate that ASD-SAENet exhibits comparable accuracy (70.8%), and superior specificity (79.1%) for the whole dataset as compared to other methods. Further, our experiments demonstrate superior results as compared to other state-of-the-art methods on 12 out of the 17 imaging centers exhibiting …
