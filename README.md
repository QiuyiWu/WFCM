# Statistical Inference for Fuzzy Clustering (WFCM) — Code Repository

This repository contains the reproducible code and data accompanying the paper “Statistical Inference for Fuzzy Clustering” by Qiuyi Wu, Zihan Zhu and Anru R. Zhang.

The code implements the proposed Weighted Fuzzy C-Means (WFCM) framework, supports uncertainty quantification, and reproduces all figures as well as simulations and applications found in the manuscript.

## Repository Structure
```
├── data/
│   ├── processed_mix3_df.rds          # REMOVED!! RNA-seq processed dataset (Example 1)
│   ├── ADNI_pc.csv                    # REMOVED!! ADNI PCA features (Example 2)
│   ├── ADNI_data.csv                  # REMOVED!! ADNI extracted features before PCA
│   ├── ADNIMERGE_0.0.1.tar.gz         # REMOVED!! Raw ADNI data (for full reproducibility)
│
├── FCM_figure.ipynb                   # Generates Figures 1 and 2 (Section 3)
├── FCM_cell.ipynb                     # RNA-seq case study (Application Example 1)
├── FCM_adni.ipynb                     # ADNI case study (Application Example 2)
└── README.md
```
> **Note:** The raw ADNI data (`ADNIMERGE_0.0.1.tar.gz`) and any individual-level ADNI CSV files are **not included** due to data use restrictions. 
> Where to Find and Access the Data: Alzheimer’s Disease Neuroimaging Initiative (ADNI)
>
> **Instructions to reproduce `ADNI_pc.csv`:**
> 
> Access process: Register for an account and submit a Data Use Application through the ADNI Image and Data Archive (IDA) Alzheimer's Journals.Upon approval, you can download imaging and biomarker datasets in CSV or DICOM formats, including preprocessed PET measures (e.g., florbetapir SUVR by region) and MRI FreeSurfer outputs.
>
> Go to tab Downloads --- Study Info --- Data&Database --- ADNIMERGE - Key ADNI tables merged into one table - Packages for R [ADNI1,GO,2]
> PCA analysis on selected features



> **Note:** The file `processed_mix3_df.rds` is derived from single-cell RNA-seq data (Zheng et al., 2017) and processed as in:
> Gao, L. L., Bien, J., & Witten, D. (2024). *Selective inference for hierarchical clustering.* Journal of the American Statistical Association, 119(545), 332–342.  
> Original RNA-seq data and R code are available at:  
> [https://github.com/lucylgao/clusterpval-experiments/blob/master/real-data-code/zheng.R](https://github.com/lucylgao/clusterpval-experiments/blob/master/real-data-code/zheng.R)
>
> **Instructions to reproduce `processed_mix3_df.rds`:**
> 
> 1. Download the Zheng et al. (2017) single-cell RNA-seq data.
> 2. Run `zheng.R` from Lucy Gao’s repository to generate the processed mixture dataset.
> 3. Save the resulting R object as `data/processed_mix3_df.rds` locally.



## Contents
1. Figures for Section 3
    - Notebook: FCM_figure.ipynb
    - Purpose: Reproduce Figure 1 and Figure 2 in Section 3
    - Includes:
        - Synthetic data generation
        - Comparison of classical FCM vs. WFCM
        - Induced density for different parameter values
2. Simulation
3. Application 
    - Example 1: RNA-seq (Single-Cell) Data
        - Notebook: `FCM_cell.ipynb`
        - Data: `data/processed_mix3_df.rds`
        - Description: Implements WFCM on a synthetic RNA-seq mixture dataset. Demonstrates clustering structure, soft memberships, and inference results.
    - Example 2: ADNI (Alzheimer's Disease) Data
        - Notebook: `FCM_adni.ipynb`
        - Data: `ADNI_pc.csv` — PCA-transformed features used in WFCM
                `ADNI_data.csv` — extracted ADNI features before PCA
                `ADNIMERGE_0.0.1.tar.gz` — raw ADNI dataset
        - Description: Performs WFCM on ADNI biomarker and clinical data. Produces all figures and results used in the ADNI application section.


## Dependencies
The notebooks rely on common scientific Python libraries, including:
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `sklearn`
- `rpy2` (optional; used if reading `.rds` files in Python)


## Reproducibility
Each notebook is self-contained and can be run independently.

All datasets required for reproducing the figures and results are included in the `data/` folder.


## Citation
If you use this code in your research, please cite:

Wu, Q., Zhu, Z., & Zhang, A. R. (2025). Statistical Inference for Fuzzy Clustering. Manuscript.

## Contact

For questions, please contact:
Qiuyi Wu (qiuyi.wu@duke.edu) or Zihan Zhu (zhzhu1@wharton.upenn.edu).


