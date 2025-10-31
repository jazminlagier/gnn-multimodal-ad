---
layout: paper
title: "Exploring the impact of muti-site data acquisition parameters on deep learning models for fMRI data"
nickname: X-AI-poster
authors: "Fahad Almuqhim; Saeed, Fahad;"
year: "2023"
conference: "AI-X-Symposium-2023"
image: /assets/images/posters/mutisite_fMRI-poster.png
projects: ["ML-brain-imaging"]
tags: []

# Content
# fulltext: https://alz.confex.com/alz/2024/meetingapp.cgi/Paper/89944
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
Multisite datasets have become increasingly popular due to their diversity and generalizability. However, variations in data acquisition parameters across sites can introduce confounding effects that challenge the reliability of machine learning (ML) models in neuroimaging studies. Using the Autism Brain Imaging Data Exchange I (ABIDE-I) functional magnetic resonance imaging (fMRI) dataset, consisting of 1035 subjects from 17 sites, we investigated the influence of site-specific factors on deep learning (DL) model performance. By evaluating two published models, ASD-DiagNet and ASD-SAENet, we demonstrated that despite preprocessing efforts, site-specific variations significantly impact classification outcomes. Our results highlight the need for robust preprocessing pipelines to mitigate these effects and ensure model generalizability for neuroimaging-based ASD diagnosis.

# Methods
The study utilized the ABIDE-I dataset, comprising 1035 subjects, including 505 with autism spectrum disorder (ASD) and 530 healthy controls (HC). The dataset spans 17 sites with diverse acquisition parameters, including scanner manufacturer, repetition time (TR), and echo time (TE). Preprocessed scans provided blood oxygen level-dependent (BOLD) signals, with brain activity represented by 200 regions of interest (ROIs) using the CC200 atlas. Pearson’s correlations of ROIs yielded 19,900 features per subject as model input. Two deep learning models were employed: ASD-DiagNet, which integrates a tied autoencoder and single-layer classifier, and ASD-SAENet, combining a sparse autoencoder with a multilayer classifier. Classification performance was evaluated for pairs of centers with varying acquisition parameters.


# Results
Our analysis revealed that deep learning models could classify centers based on site-specific variations in scanning parameters with high accuracy. Across all centers, the models achieved 97.57% accuracy, 99.93% sensitivity, and 96.55% specificity. When classifying centers using different repetition times (TRs), performance remained robust, with 89.95% accuracy, 97.33% sensitivity, and 87.95% specificity. These results demonstrate that site-specific acquisition parameters significantly impact classification performance, suggesting that the models may inadvertently learn site-related biases instead of ASD-relevant biomarkers.


# Conclusions
Our findings underscore the persistent challenge of site-specific effects in multisite neuroimaging datasets like ABIDE-I, even after employing standard preprocessing pipelines. These effects compromise the ability of deep learning models to accurately detect ASD biomarkers, as they may primarily learn site-related variations. To enhance model generalizability and reliability, future work should focus on developing advanced preprocessing techniques and strategies to minimize site-specific biases, ensuring that ML models capture meaningful biological patterns relevant to ASD.
