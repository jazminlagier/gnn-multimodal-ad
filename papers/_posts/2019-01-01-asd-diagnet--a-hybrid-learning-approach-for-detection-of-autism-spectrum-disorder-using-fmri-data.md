---
layout: paper
title: "ASD-DiagNet: A hybrid learning approach for detection of Autism Spectrum Disorder using fMRI data"
nickname: 2019-01-01-asd-diagnet--a-hybrid-learning-approach-for-detection-of-autism-spectrum-disorder-using-fmri-data
authors: "Eslami, Taban; Mirjalili, Vahid; Fong, Alvis; Laird, Angela; Saeed, Fahad; "
year: "2019"
journal: "Frontiers Frontiers in Neuroinformatics"
volume: 13
issue:
pages: 70
is_published: True
image: /assets/images/papers/biorxiv.png
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
doi: "10.3389/fninf.2019.00070/full"
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

Heterogeneous mental disorders such as Autism Spectrum Disorder (ASD) are notoriously difficult to diagnose, especially in children. The current psychiatric diagnostic process is based purely on the behavioral observation of symptomology (DSM-5/ICD-10) and may be prone to misdiagnosis. In order to move the field toward more quantitative diagnosis, we need advanced and scalable machine learning infrastructure that will allow us to identify reliable biomarkers of mental health disorders. In this paper, we propose a framework called ASD-DiagNet for classifying subjects with ASD from healthy subjects by using only fMRI data. We designed and implemented a joint learning procedure using an autoencoder and a single layer perceptron (SLP) which results in improved quality of extracted features and optimized parameters for the model. Further, we designed and implemented a data augmentation strategy, based on linear interpolation on available feature vectors, that allows us to produce synthetic datasets needed for training of machine learning models. The proposed approach is evaluated on a public dataset provided by Autism Brain Imaging Data Exchange including 1, 035 subjects coming from 17 different brain imaging centers. Our machine learning model outperforms other state of the art methods from 10 imaging centers with increase in classification accuracy up to 28% with maximum accuracy of 82%. The machine learning technique presented in this paper, in addition to yielding better quality, gives enormous advantages in terms of execution time (40 min vs. 7 h on other methods). The implemented code is available as GPL license …
