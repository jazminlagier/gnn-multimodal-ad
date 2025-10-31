---
layout: paper
title: "Predicting Epileptic Seizures using Large Language Models"
nickname: 3MT-2024-presentation
authors: "Parani, Paras; Mohammad, Umair; Saeed, Fahad; "
year: "2024"
conference: ""
image: /assets/images/posters/3MT-2024-presentation.jpg
projects: [ML-seizure]
tags: []

# Content
fulltext:
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
Epileptic seizures affect over 50 million people worldwide, often with little to no warning, leading to severe health risks and diminished quality of life. Traditional methods for seizure prediction face challenges in achieving generalizability across diverse patient data. This study leverages Large Language Models (LLMs) and domain adaptation techniques to predict epileptic seizures up to 5 minutes in advance using patient-independent EEG data. By analyzing patterns in EEG recordings, the model achieves an approximate accuracy of 74%, offering a promising step towards real-time, early detection systems. Future advancements aim to refine the model for enhanced accuracy and scalability.

# Methods
We leveraged Large Language Models (LLMs) pre-trained on textual data and adapted them for patient-independent epileptic seizure prediction using EEG time-series data. The EEG data was transformed into a format compatible with the LLM architecture by tokenizing individual channels and generating sequences of string tokens, reducing memory requirements. The model used a sliding attention mechanism and global attention layers to process sequences up to 4,096 tokens effectively. The classification task involved predicting preictal or interictal phases using majority voting across 20 channels.

Additionally, we integrated domain adaptation techniques to mitigate data variability, ensuring robust generalization across patients. Hyperparameter optimization included a learning rate of 5e-4, a warm-up ratio of 0.03, weight decay of 1%, and training for 4 epochs. The training process utilized high-performance computing resources to expedite experimentation while maintaining efficiency.

# Results
The adapted LLM model demonstrated strong performance, achieving an approximate accuracy of 74% for predicting seizures up to 5 minutes in advance. Sensitivity and specificity were balanced to optimize detection of preictal phases while minimizing false alarms. The confusion matrix revealed 6,407 true positives, 4,739 true negatives, 2,766 false negatives, and 1,088 false positives. Compared to other deep learning approaches such as ViTs and ResNets, the LLM provided superior predictive capabilities, making it a robust tool for seizure prediction. Future work aims to enhance model interpretability, integrate multi-modal data, and enable real-time deployment for clinical applications. 

# Conclusions
The presented model demonstrates the viability of using LLMs and domain adaptation techniques for early seizure prediction. Its ability to predict seizures up to 5 minutes before onset highlights its potential to transform epilepsy care by offering patients critical time for preventive actions. Future developments will focus on improving model sensitivity, integrating multi-modal inputs, and enabling scalable real-time applications. This study represents a significant advancement in the use of AI for personalized healthcare and opens avenues for broader applications in medical time-series analysis.