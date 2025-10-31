---
layout: paper
title: "PVTADNet: Pyramid Vision Transformer to Diagnose Alzheimer’s Disease Using structural MRI Data"
nickname: KFSCIS-ma-2023-poster
authors: "Maryam Akhavan Aghdam; Bozdag, Serdar; Saeed, Fahad;"
year: "2023"
conference: "KFSCIS-2023"
image: /assets/images/posters/PVTADNet-poster.png
projects: ["ML-brain-imaging", "ML-ADRD"]
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
Alzheimer's disease (AD) is a neurodegenerative disorder and timely diagnosis is crucial for early interventions. AD has disruptive local and global brain neural connections that may be instrumental in understanding and extracting specific biomarkers. Previous machine-learning approaches are mostly based on convolutional neural network (CNN) and standard vision transformer (ViT) models which may not sufficiently capture the multidimensional local and global patterns that may be indicative of AD. We propose a novel approach called PVTADNet to classify AD and cognitively normal (CN) cases using pretrained pyramid vision transformer (PVT) and white matter (WM) of T1-weighted structural MRI (sMRI) data. Our approach combines the advantages of CNN and standard ViT to extract both local and global features indicative of AD from the WM coronal middle slices. We performed experiments on subjects with T1-weighed sMRI scans from the ADNI dataset. PVTADNet outperforms the single and parallel CNN and standard ViT architectures based on sMRI data for AD vs. CN classification.

# Methods
We utilized the PVT model to analyze WM coronal middle slices. PVT is an extension of ViT architecture with a hierarchical (pyramid) feature extractor. PVT uses multiple levels of transformers at different scales to capture both local and global features of WM coronal middle slices in a unified architecture. The model processes sMRI coronal middle slices as a sequence of variable-size patches, which are linearly embedded. PVT consists of multiple stages, each extracting features at different resolution levels. Figure 3 illustrates the four-stage process of PVT that produces feature maps of coronal slices at different scales. Each stage has an architecture consisting of a patch embedding layer and 𝐿𝑖 transformer encoder layers.

# Results
The PVTAD model outperforms existing methods for Alzheimer's classification, achieving the highest accuracy (97.7%), sensitivity (97.15%), specificity (98.16%), precision (98.02%), and F1-Score (97.6%) using sMRI (WM Coronal View) data. It surpasses ResNet-18, ViT-Tiny, and their parallel combination, demonstrating its effectiveness in leveraging white matter features for improved performance.

# Conclusions
The proposed PVTAD model demonstrates exceptional performance in classifying Alzheimer's disease (AD) compared to existing methods, achieving the highest accuracy, sensitivity, specificity, precision, and F1-Score. By leveraging white matter features from sMRI (WM Coronal View) data and integrating the strengths of CNNs and vision transformers, the model effectively captures both local and global patterns indicative of AD. These results highlight the potential of the PVTAD model as a reliable and robust tool for early diagnosis of AD, offering significant improvements over traditional approaches.
