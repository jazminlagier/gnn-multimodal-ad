---
layout: paper
title: "SpeCollate: Deep cross-modal similarity network for mass spectrometry data based peptide deductions"
nickname: 2021-01-01-specollate--deep-cross-modal-similarity-network-for-mass-spectrometry-data-based-peptide-deductions
authors: "Tariq, Muhammad Usman; Saeed, Fahad; "
year: "2021"
journal: "Public Library of Science San Francisco, CA USA PloS one"
volume: 16
issue:
pages: e0259349
is_published: True
image: /assets/images/papers/plos.png
projects: [ML-MS]
tags: []

# Text
fulltext:
pdf:
pdflink:
pmcid:
preprint: 
supplement:

# Links
doi: "10.1371/journal.pone.0259349"
pmid:

# Data and code
github: []
neurovault:
openneuro: []
figshare:
figshare_names:
osf:
---
{% include JB/setup %}

# Abstract

Historically, the database search algorithms have been the de facto standard for inferring peptides from mass spectrometry (MS) data. Database search algorithms deduce peptides by transforming theoretical peptides into theoretical spectra and matching them to the experimental spectra. Heuristic similarity-scoring functions are used to match an experimental spectrum to a theoretical spectrum. However, the heuristic nature of the scoring functions and the simple transformation of the peptides into theoretical spectra, along with noisy mass spectra for the less abundant peptides, can introduce a cascade of inaccuracies. In this paper, we design and implement a Deep Cross-Modal Similarity Network called SpeCollate, which overcomes these inaccuracies by learning the similarity function between experimental spectra and peptides directly from the labeled MS data. SpeCollate transforms spectra and peptides into a shared Euclidean subspace by learning fixed size embeddings for both. Our proposed deep-learning network trains on sextuplets of positive and negative examples coupled with our custom-designed SNAP-loss function. Online hardest negative mining is used to select the appropriate negative examples for optimal training performance. We use 4.8 million sextuplets obtained from the NIST and MassIVE peptide libraries to train the network and demonstrate that for closed search, SpeCollate is able to perform better than Crux and MSFragger in terms of the number of peptide-spectrum matches (PSMs) and unique peptides identified under 1% FDR for real-world data. SpeCollate also identifies a large number of peptides not reported by …
