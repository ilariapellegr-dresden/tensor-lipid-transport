# Tensor analysis for lipid transport

Code for *Tensor analysis for lipid transport* by Ilaria Pellegrinelli, Vincenzo Galgano, Mathilda Lennartz, André Nadler and Heather A. Harrington (MPI-CBG, Dresden).

We look at a lipid transport dataset as a tensor of order three: 8 lipid probes × 5 organelles × 5 time points. Each entry of the tensor contains the mean abundance of a lipid in an organelle at a given time, from the experiments of Iglesias-Artola et al. 

The analysis is split into two scripts:

- `CP_bio_data.py`: empirically optimal consensus-based rank selection, non-negative CP decomposition and the binary mask for missing values
- `HOSVD_bio_data.py`: HOSVD (full, truncated) and the empirical confidence bounds for the bootstrapping approach (noise).

## How the pipeline works

The pipeline follows the steps of Figure 2 of the paper. Both scripts first build the tensor from `SummaryData.csv`, keeping only UV-irradiated samples (UV = 1) and excluding the lipid probe KB27. Each lipid slice is divided by its Frobenius norm, so that lipids with very different abundances can be compared.


## Citation

If you use this code, please cite:

Pellegrinelli I, Galgano V, Lennartz M, Nadler A, Harrington HA. Tensor analysis for lipid transport. 
