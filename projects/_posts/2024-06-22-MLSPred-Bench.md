---
layout: project
title: "MLSPred-Bench: Reference EEG Benchmark for Prediction of Epileptic Seizures"
contributors: [Prof-S,umairmy305]
handle: MLSPred-Bench
status: collection
type: dataset

# Optional
website: 
grant: [{id: "NSF TT-2213951", url: "https://www.nsf.gov/awardsearch/showAward?AWD_ID=2213951&HistoricalAwards=false"}]
image: /assets/images/projects/2024-06-22-raw-EEG-signal.png
tagline: 'Reference EEG Benchmark for Prediction of Epileptic Seizures'
tags: [eeg]

# Data and code
github: https://github.com/pcdslab/MLSPred-Bench
neurovault:
openneuro:
figshare:
figshare_names:
osf:
---
{% include JB/setup %}

MLSPred-Bench will create 12 different benchmarks based on different values of the seizure prediction horizon (SPH) and the seizure occurrence period (SOP). The benchmarks are for patient-independent epileptic seizure prediction using only raw electroencephalography (EEG) data and are machine learning (ML)-ready.

For each benchmark, MLSPred-Bench draws preictal segments of length from the SPH duration. We assume there is a gap equal to the SOP in minutes before the start of a seizure where the SPH ends. The datasets are class-balanced where an equal amount of interictal samples are drawn from sessions of the same subject where there were no seizures. 


We designed and developed a method called MLSPred-Bench that can be used for converting any EEG big data annotated for detection into ML-ready data suitable for prediction. We apply our methods to the existing EEG data corpus to generate 12 ML-ready benchmarks comprising data for training, validating, and testing seizure prediction models. Our strategy uses different variations of seizure prediction horizon (SPH) and the seizure occurrence period (SOP) to produce more than 150GB of ML-ready data. We hope that the generated benchmarking data will be utilized by various computational groups for their seizure prediction model development.

Send an email at fsaeed@fiu.edu if you want to get pre-processed MLSPred-Bench data. 

The work can be summarized as follows:
1. Extract short preictal and interictal segments from long-duration annotated EEG montages.
2. Generate a comprehensive list of ML-ready benchmarks with varying SPH and SOP.
3. Technically validate the generated data with multiple ML and DL models with up-to 88.73% validation accuracy
4. Opensource code and related materials are available at https://github.com/pcdslab/MLSPred-Bench.




## Participate

Data collection and curation for this study is complete.
