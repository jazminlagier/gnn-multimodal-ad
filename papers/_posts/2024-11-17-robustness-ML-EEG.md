---
layout: paper
title: "Robustness of ML-Based Seizure Prediction Using Noisy EEG Data From Limited Channels"
nickname: Robustness ML EEG
authors: "Mohammad, Umair; Saeed, Fahad; "
year: "2024"
journal: "Proceedings of 20th International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)"
volume: 
issue:
pages: 620-626
is_published: True
image: /assets/images/papers/ieee.png
projects: [ML-seizure]
tags: [workshop]

# Text
fulltext: 
pdf: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10621571
pdflink:
pmcid:
preprint: 
supplement:

# Links
doi: 10.1109/DCOSS-IoT61029.2024.00097
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

Seizures pose a significant health hazard for over 50 million individuals with epilepsy worldwide, with approximately 56% experiencing uncontrollable seizures according to the CDC. Predicting seizures is challenging even with the availability of various sensors (gyroscopes, pulse rate sensors, heart rate monitors, etc). Electroencephalography (EEG) data can directly measure the activity of the brain and has been the choice of leveraging deep learning (DL) models for seizure prediction. Despite DL models achieving over 95% accuracy on retroactive clinical-grade EEG data, this performance fails to translate in real-world settings where the accuracy goes down to 66% which warrants further investigation. Moreover, consumer-grade wearable EEG headsets, characterized by lower data quality and a varying number of channels across brands, present additional challenges. In this paper, we estimate the robustness of DL models which are trained on clinical-grade EEG data but tested on the type of data expected from consumer-grade wearable EEG headsets. We select the previously published model SPERTL to estimate its robustness when: (1) predicting with data from less leads/channels, (2) predicting when faced with streaming data, (3) evaluating performance on imbalanced data with more interictal segments. Our results are compared against baseline results from the SPERTL model which we have re-configured to operate independently of the number of channels with an average baseline area under the curve (AUC) score of 98.56%. Our results demonstrate that though the model is surprisingly resilient to streaming and noisy data, reducing the number of channels and a higher class imbalance have a more severe degradation. The AUC across all cross-validation sets degrades only by 2% and 3% on average for noisy and streaming data, respectively. However, a performance reduction, on average, is observed by 32% when imbalance is increased with higher percentage of interictal samples, and up to 16% when using lower number of channels.