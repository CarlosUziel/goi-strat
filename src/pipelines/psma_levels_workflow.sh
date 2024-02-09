#!/bin/bash
# run from ~/biopipes/src/pipelines

## PCTA-WCDT
python data/run/rna_seq/pcta_wcdt/generate_gsva.py && \
python data/run/rna_seq/pcta_wcdt/goi_gsva_splits_annotation.py && \
python differential_expression/run/pcta_wcdt/goi.py && \
python functional_analysis/run/rna_seq/pcta_wcdt/goi.py && \
python differential_enrichment/run/pcta_wcdt/goi.py

## TCGA-PRAD
python data/run/meth_array/tcga_prad/generate_gsva.py && \
python data/run/meth_array/tcga_prad/goi_gsva_splits_annotation.py && \
python differential_methylation/run/tcga_prad/goi.py && \
python functional_analysis/run/meth_array/tcga_prad/goi.py

## WCDT-MCRPC
python data/run/rrbs/wcdt/generate_gsva.py && \
python data/run/rrbs/wcdt/goi_gsva_splits_annotation.py && \
python differential_methylation/run/wcdt/goi_methylkit.py && \
python functional_analysis/run/rrbs/wcdt/goi_methylkit.py

## Integrative analysis
python integrative_analysis/run/rna_seq/pcta_wcdt/goi_intersect_degss_gsea.py && \
python degss_genes_stats/run/rna_seq/pcta_wcdt/goi.py && \
python degss_genes_stats/run/rna_seq/pcta_wcdt/goi_metadata.py && \
python degss_ppi_networks/run/rna_seq/pcta_wcdt/goi.py && \
python degss_ppi_networks_clustering/run/rna_seq/pcta_wcdt/goi.py && \
python functional_analysis/run/rna_seq/pcta_wcdt/goi_clustering_ora.py
