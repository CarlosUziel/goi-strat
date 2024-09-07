#!/bin/bash
# run from ~/goi-strat/src/pipelines

## 1. Download, process and split data
python data/run/rna_seq/multi_dataset/download_data.py --threads 8 && \
python data/run/rna_seq/multi_dataset/generate_gsva.py --threads 8 && \
python data/run/rna_seq/multi_dataset/goi_gsva_splits_annotation.py && \
python data/run/rna_seq/multi_dataset/goi_perc_splits_annotation.py

## 2. Differential analyses
python differential_expression/run/multi_dataset/goi_gsva.py --threads 16 && \
python differential_expression/run/multi_dataset/goi_perc.py --threads 16 && \
python differential_enrichment/run/multi_dataset/goi_gsva.py --threads 16 && \
python differential_enrichment/run/multi_dataset/goi_perc.py --threads 16