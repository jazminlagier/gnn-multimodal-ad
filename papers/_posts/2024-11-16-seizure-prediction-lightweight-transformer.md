---
layout: paper
title: "Lightweight Transformer exhibits comparable performance to LLMs for Seizure Prediction: A case for light-weight models for EEG data"
nickname: ESPFormer for Seizure Prediction
authors: "Parani, Paras; Mohammad, Umair; Saeed, Fahad; "
year: "2024"
journal: "Proceedings of the IEEE International Conference on Big Data (IEEE BigData)"
volume: 
issue:
pages: 4941-4945
is_published: True
image: /assets/images/papers/ieee.png
projects: [ML-seizure]
tags: [workshop]

# Text
fulltext: https://ieeexplore.ieee.org/document/10825319
pdf: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10825319
pdflink:
pmcid: PMC11877310  
preprint: 
supplement:

# Links
doi: 10.1109/BigData62323.2024.10825319
pmid:

# Data and code
github: 
neurovault:
openneuro: []
figshare:
figshare_names:
osf:
---
{% include JB/setup %}

# Abstract

Predicting seizures ahead of time will have a significant positive clinical impact for people with epilepsy. Advances in machine learning/artificial intelligence (ML/AI) has provided us the tools needed to perform such predictive tasks. To date, advanced deep learning (DL) architectures such as the convolutional neural network (CNN) and long short-term memory (LSTM) have been used with mixed results. However, highly connected activity exhibited by epileptic seizures necessitates the design of more complex ML techniques which can better capture the complex interconnected neurological processes. Other challenges include the variability of EEG sensor data quality, different epilepsy and seizure profiles, lack of annotated datasets and absence of ML-ready benchmarks. In addition, successful models will need to perform inference in almost real-time using limited hardware compute-capacity. To address these challenges, we propose a lightweight architecture, called ESPFormer, whose novelty lies in the simple and smaller model-size and a lower computational load footprint needed to infer in real-time compared to other works in the literature. To quantify the performance of this lightweight model, we compared its performance with a custom-designed residual neural network (ResNet), a pre-trained vision transformer (ViT) and a pre-trained large-language model (LLM). We tested ESPFormer on MLSPred-Bench which is the largest patient-independent seizure prediction dataset comprising 12 benchmarks. Our results demonstrate that ESPFormer provides the best performance in terms of prediction accuracy for 4/12 benchmarks with an average improvement of 2.65% compared to the LLM, 3.35% compared to the ViT and 17.65% compared to the ResNet – and comparable results for other benchmarks. Our results indicate that lightweight transformer architecture may outperform resource-intensive LLM based models for real-time EEG-based seizure predictions.