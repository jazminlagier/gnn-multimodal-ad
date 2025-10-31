---
layout: paper
title: "Utilizing Pretrained Vision Transfomers and Large Language Models for Epileptic Seizure Prediction"
nickname: ViTs and LLMs for Seizure Prediction
authors: "Parani, Paras; Mohammad, Umair; Saeed, Fahad; "
year: "2025"
journal: "Proceedings of the 8th International Conference on Data Science and Machine Learning (CDMA 2025)"
volume: 
issue:
pages: 132-137
is_published: True
image: /assets/images/papers/ieee.png
projects: [ML-seizure]
tags: [conference]

# Text
fulltext: https://ieeexplore.ieee.org/abstract/document/10908748
pdf: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10908748
pdflink:
pmcid:
preprint: https://www.biorxiv.org/content/10.1101/2024.11.03.621742v1
supplement:

# Links
doi: "10.1109/CDMA61895.2025.00028"
pmid:

# Data and code
github: [https://github.com/pcdslab/UtilLLM_EPS]
neurovault:
openneuro: []
figshare:
figshare_names:
osf:
---
{% include JB/setup %}

# Abstract

Repeated unprovoked seizures due to epilepsy is a major source of concern for patients suffering from this debilitating condition. Predicting these seizures before they occur is an active area of research and interest to both machine-learning (ML) scientists as well as clinicians. The variability of EEG data, and different kinds of seizures present significant challenges, complicating the large-scale annotation of data for predictive applications. To address these challenges, we propose the use of pre-trained models, such as Vision Transformers (ViTs) and Large Language Models (LLMs), which were  trained using publicly available image or text data, but have not been extensively developed or tested across diverse EEG datasets for predicting seizures. This work leverages pre-trained ViTs and LLMs by minimalistic retraining and refining of the input, embedding, and classifications layers and reports the results for seizure prediction. Our results demonstrate that the LLM outperforms the ViTs in patient-independent seizure prediction by achieving a sensitivity of 79.02\% which is 8\% higher compared to ViTs and about 12\% higher compared to a custom-designed ResNet-based model. This work represents a significant step forward in studying the feasibility of pre-trained models for seizure prediction with its potential for improving the quality of life of people with epilepsy. Our code and related materials are available open-source at: https://github.com/pcdslab/UtilLLM_EPS/
