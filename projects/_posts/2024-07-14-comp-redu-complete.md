---
layout: project
title: Compressive and reductive analysis of genomic and proteomics data
contributors: [mgawan, sandino, Prof-S]
handle:
status: complete
type: analysis


# Optional
website:
grant: [{id: "NSF 1464268", url: "https://www.nsf.gov/awardsearch/showAward?AWD_ID=1464268&HistoricalAwards=false"}]
image: /assets/images/projects/tpds-gagraphic-2692782.jpg
tagline: Development of methods that can operate on compressed big data
tags: []

# Data and code
github: ["https://github.com/pcdslab/MSREDUCE","https://github.com/pcdslab/PHYNGSC"]
neurovault:
openneuro:
figshare:
figshare_names:
osf:
---
{% include JB/setup %}

Sequencing of genomes for numerous species including humans has become increasingly affordable due to next generation high-throughput genome sequencing (NGS) technologies and large-scale mass spectrometry data. This opens up perspectives for diagnosis and treatment of genetic diseases and is increasingly effective in conducting system biology studies. However, there remain many computational challenges that need to be addressed before these technologies find their way into every day health and human care. One such daunting challenge is the volume of sequencing and MS data which can reach peta-byte level for comprehensive system-biology studies.

Genomic data compression is needed to reduce the storage size, to increase the speed and reduce the cost of I/O bandwidth required for transmission of such data. However, existing genomic compression solutions yield poor performance for Big Genomic Data. Further, the existing state of the art tools require the user to decompress the data before it can be used for further analysis. This project is focused on compression of genomic information and developing a framework which will allow analysis of compressed form of the data. The project develops HPC solutions for fast compression of Big NGS Data sets using ubiquitous architectures such as GPUs and multicore processors. HPC techniques are utilized to compute essential functions such as alignment and mapping using the compressed form of the NGS data. More efficient encoding of the NGS data for better network utilization is also being investigated.

Similarly the project develops reductive and compressive techniques to process mass spectrometry data. 


## Status

The method development of this work is complete. 