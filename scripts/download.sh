#!/bin/sh
# Public data only (NCBI GEO). ~3.5 GB. NOTE: do not download the platform's *family* SOFT file (135 GB, every sample
# ever run on EPIC); the probe manifest is the 161 MB supplementary CSV below.
set -e
mkdir -p data/raw && cd data/raw
G=https://ftp.ncbi.nlm.nih.gov/geo
curl -fO $G/series/GSE220nnn/GSE220622/matrix/GSE220622_series_matrix.txt.gz
curl -fO $G/series/GSE220nnn/GSE220622/suppl/GSE220622_processed_M-value.tsv.gz
curl -fO $G/series/GSE220nnn/GSE220622/suppl/GSE220622_signal_intensities_matrix.tsv.gz
curl -f -o GPL21145_MethylationEPIC_manifest.csv.gz $G/platforms/GPL21nnn/GPL21145/suppl/GPL21145_MethylationEPIC_15073387_v-1-0.csv.gz
