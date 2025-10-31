---
layout: project
title: UtilLLM_EPS
contributors: [Paras1242, umairmy305, Prof-S]
handle: UtilLLM_EPS
status: complete


# Optional
website: 
grant:
grant_url:
image: /assets/images/software/open-source.png
tagline: Using pretrained LLM and ViTs to Predict Epileptic Seizure 
tags: [software]

# Data and code
github: https://github.com/pcdslab/UtilLLM_EPS
neurovault:
openneuro:
figshare:
figshare_names:
osf:
---
{% include JB/setup %}

we propose the use of pre-trained models, such as Vision Transformers (ViTs) and Large Language Models (LLMs), which were  trained using publicly available image or text data, but have not been extensively developed or tested across diverse EEG datasets for predicting seizures. This work leverages pre-trained ViTs and LLMs by minimalistic retraining and refining of the input, embedding, and classifications layers and reports the results for seizure prediction. Our results demonstrate that the LLM outperforms the ViTs in patient-independent seizure prediction by achieving a sensitivity of 79.02\% which is 8\% higher compared to ViTs and about 12\% higher compared to a custom-designed ResNet-based model. This work represents a significant step forward in studying the feasibility of pre-trained models for seizure prediction with its potential for improving the quality of life of people with epilepsy.

## Status 
The method development for this work is ongoing.