---
layout: paper
title: "Alzheimer’s Disease Diagnosis Using Gray Matter of T1-Weighted sMRI Data and Vision Transformer"
nickname: AAIC-2024-poster
authors: "Maryam Akhavan Aghdam; Bozdag, Serdar; Saeed, Fahad;"
year: "2024"
conference: "AAIC-2024"
image: /assets/images/posters/AAIC-2024-poster.png
projects: ["ML-brain-imaging", "ML-ADRD"]
tags: []

# Content
fulltext: https://alz.confex.com/alz/2024/meetingapp.cgi/Paper/89944
pdf: 

# Links
doi: 


# Data and code
github:
neurovault:
openneuro:
figshare:
figshare_names:
osf:
f1000:
---

{% include JB/setup %}

# Abstract
Traditional Alzheimer's Disease (AD) diagnostic methods, mainly based on cognitive, memory, and behavioral tests, have limitations, particularly in the early detection of AD. Structural magnetic resonance imaging (sMRI) has emerged as a key tool in understanding the brain changes associated with AD, focusing particularly on alterations in gray matter (GM). In recent years, machine learning (ML) models have shown great potential in interpreting complex neuroimaging data. These models can detect intricate patterns in neuroimaging data, making them invaluable in enhancing the diagnostic accuracy and early AD diagnosis. 

# Methods
We propose a novel approach to diagnose AD using Vision Transformer (ViT) and GM of T1-weighted sMRI data. The proposed approach leverages the power of deep-learning (DL) to detect the GM changes that are indicative of AD. We used pretrained ViT model to extract features from the GM sagittal and coronal slices of sMRI data and classify AD from cognitively normal (CN) as shown in Fig 1. We employed ADNI dataset, focusing on subjects with T1-weighed MPRAGE sMRI scans, including 70 AD patients and 85 CN individuals. 

# Results
The study achieved an average classification accuracy of 97.6% in sagittal slices and 97.7% in coronal slices (Figures 2 and 3). The high degree of sensitivity (96.2% to 98%) and specificity (97.1% and 98%) along with excellent ROC/AUC curves (Fig. 3) indicate that the proposed method has superior performance metrics. Comparison of the proposed method with deep-learning models, including CNN, LSTM and different combinations of CNN+LSTM indicates that the ViT transformer architecture can learn the global and local AD signatures at a superior rate than other state of the art methods. 

# Conclusions
Proposed Transformer architecture can learn both the global and local AD signatures at a superior rate than the other state of the art methods. The proposed approach demonstrates a significant advancement in the accurate diagnosis of AD, particularly enhancing its early diagnosis. It represents a considerable improvement over existing diagnostic methods, offering a new avenue for research and application in the field of neurodegenerative diseases. The future work consists of designing and developing more domain specific machine-learning models that can diagnose and characterize healthy controls, mild cognitive impairments (MCI) and Alzheimer's disease scans. Such models can be used for characterizing and predicting different AD and related dementias. 