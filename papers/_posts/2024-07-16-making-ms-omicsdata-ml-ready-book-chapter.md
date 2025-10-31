---
layout: paper
title: "Making MS Omics Data ML-Ready: SpeCollate Protocols"
nickname: 2024-07-16-SpeCollate-Protocols
authors: "Tariq,Muhammad Usman; Ebert, Samuel; Saeed, Fahad;"
year: "2024"
journal: "Springer Protein Bioinformatics"
volume: 
issue:
pages: 142-162
is_published: True
image: /assets/images/papers/Protein_bio.png
projects: [ML-MS]
tags: []

# Text
fulltext: https://link.springer.com/protocol/10.1007/978-1-0716-4007-4_9
pdf: /assets/pdfs/Protein_Book-chapter-2024.pdf
pdflink:
pmcid:
preprint: 
supplement:

# Links
doi: "10.1007/978-1-0716-4007-4_9"
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

The increasing complexity and volume of mass spectrometry (MS) data have presented new challenges and opportunities for proteomics data analysis and interpretation. In this chapter, we provide a comprehensive guide to transforming MS data for machine learning (ML) training, inference, and applications. The chapter is organized into three parts. The first part describes the data analysis needed for MS-based experiments and a general introduction to our deep learning model SpeCollate—which we will use throughout the chapter for illustration. The second part of the chapter explores the transformation of MS data for inference, providing a step-by-step guide for users to deduce peptides from their MS data. This section aims to bridge the gap between data acquisition and practical applications by detailing the necessary steps for data preparation and interpretation. In the final part, we present a demonstrative example of SpeCollate, a deep learning-based peptide database search engine that overcomes the problems of simplistic simulation of theoretical spectra and heuristic scoring functions for peptide-spectrum matches by generating joint embeddings for spectra and peptides. SpeCollate is a user-friendly tool with an intuitive command-line interface to perform the search, showcasing the effectiveness of the techniques and methodologies discussed in the earlier sections and highlighting the potential of machine learning in the context of mass spectrometry data analysis. By offering a comprehensive overview of data transformation, inference, and ML model applications for mass spectrometry, this chapter aims to empower researchers and practitioners in leveraging the power of machine learning to unlock novel insights and drive innovation in the field of mass spectrometry-based omics.