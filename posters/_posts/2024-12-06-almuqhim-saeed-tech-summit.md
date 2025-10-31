---
layout: paper
title: "Advancing fMRI Classification of Neurological Disorders with a Novel Self-Supervised Learning Approach"
nickname: KFSCIS-fa-2024-poster
authors: "Fahad Almuqhim; Saeed, Fahad;"
year: "2024"
conference: "KFSCIS-2024"
image: /assets/images/posters/ssl_tech_sumit-poster.png
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
Neuroimaging data, particularly fMRI, presents significant challenges due to its high dimensionality and the scarcity of labeled samples. This study introduces a novel self-supervised learning framework designed to address these issues, focusing on classifying neurological disorders such as Autism Spectrum Disorder (ASD). The framework utilizes a region-wise input strategy to extract robust features from unlabeled fMRI data, overcoming site-specific variability and enhancing generalizability. Our approach achieved 70.19% classification accuracy for ASD, surpassing state-of-the-art models like BrainNPT and BYOL. These results highlight the potential of self-supervised learning to advance neuroimaging research and improve clinical applications.

# Methods
This study employs a two-phase self-supervised learning framework. In the pre-training phase, the model leverages unlabeled data from large neuroimaging datasets, such as ADHD-200 and ADNI, to learn generalizable features. A region-wise input strategy was applied, ensuring that features extracted from specific brain regions were robust and consistent across sites. In the fine-tuning phase, labeled data from the ABIDE-I dataset was used to refine the model for ASD classification. The framework incorporates a contrastive learning approach to optimize feature representation, enhancing its performance on downstream classification tasks.

# Results
The self-supervised framework demonstrated superior performance in classifying ASD compared to existing models. It achieved a classification accuracy of 70.19%, outperforming BrainNPT (62.5%) and BYOL (68.9%), which are state-of-the-art self-supervised models. Site-specific analyses further validated the model’s ability to generalize across diverse neuroimaging datasets. The region-wise input strategy proved effective in mitigating site-specific variability, leading to improved feature quality and classification accuracy. These results establish the framework’s utility in handling high-dimensional fMRI data with limited labeled samples.

# Conclusions
This study highlights the potential of self-supervised learning to transform neuroimaging data analysis, particularly for high-dimensional fMRI datasets with limited labeled samples. By leveraging unlabeled data and incorporating a region-wise input strategy, the framework significantly improved classification performance for ASD. The results demonstrate the model’s ability to generalize across diverse sites, addressing critical challenges in multi-site neuroimaging studies. This work paves the way for future applications of self-supervised learning in neuroimaging, offering a robust approach to understanding and diagnosing neurological disorders.