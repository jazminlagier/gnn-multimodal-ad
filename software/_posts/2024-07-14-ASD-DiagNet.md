---
layout: project
title: ASD-DiagNet
contributors: [Prof-S, taban]
handle: DiagNet
status: complete


# Optional
website: 
grant:
grant_url:
image: /assets/images/software/open-source.png
tagline: ML algorithm identifying Austin and atypical brains using fMRI data
tags: [software]

# Data and code
github: https://github.com/pcdslab/ASD-DiagNet
neurovault:
openneuro:
figshare:
figshare_names:
osf:
---
{% include JB/setup %}

We propose a framework called *ASD-DiagNet* for classifying subjects with Austism Spectrum Disorder (ASD) from healthy subjects by using only fMRI data. We designed and implemented a joint learning procedure using an autoencoder and a single layer perceptron which results in improved quality of extracted features and optimized parameters for the model. Further, we designed and implemented a data augmentation strategy, based on linear interpolation on available feature vectors, that allows us to produce synthetic datasets needed for training of machine learning models. The proposed approach is evaluated on a public dataset provided by Autism Brain Imaging Data Exchange including 1035 subjects coming from 17 different brain imaging centers. Our machine learning model outperforms other state of the art methods from 13 imaging centers with increase in classification accuracy up to 20% with maximum accuracy of 80%. The machine learning technique presented in this paper, in addition to yielding better quality, gives enormous advantages in terms of execution time (40 minutes vs. 6 hours on other methods).

## Status 
The method development for this work is complete