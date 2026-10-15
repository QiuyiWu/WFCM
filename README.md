# Statistical Inference for Fuzzy Clustering (WFCM) — Code Repository

This repository contains the reproducible code and data accompanying the paper “Statistical Inference for Fuzzy Clustering” by Qiuyi Wu, Zihan Zhu and Anru R. Zhang.

The code implements the proposed Weighted Fuzzy C-Means (WFCM) framework, supports uncertainty quantification, and reproduces all figures as well as simulations and applications found in the manuscript.

Revision simulations, saved results, and commands for rebuilding the comparison,
inference, and computation summaries are in [`revision_simulations/`](revision_simulations/).

## Repository Structure
```
├── data/
│   ├── processed_mix3_df.rds          # REMOVED!! RNA-seq processed dataset (Example 1)
│   ├── ADNI_pc.csv                    # REMOVED!! ADNI PCA features (Example 2)
│   ├── ADNI_data.csv                  # REMOVED!! ADNI extracted features before PCA
│   ├── ADNIMERGE_0.0.1.tar.gz         # REMOVED!! Raw ADNI data (for full reproducibility)
│
├── FCM_figure.ipynb                   # Generates Figures 1 and 2 (Section 3)
├── FCM_weight.ipynb                   # Simulation study (Section 6)
├── FCM_cell.ipynb                     # RNA-seq case study (Application Example 1)
├── FCM_adni.ipynb                     # Revised ADNI analysis, including likelihood-ratio test
├── TCGA_BRCA_three_methods.ipynb      # Revision: K-means, FCM, and WFCM on TCGA-BRCA
├── requirements-applications.txt     # Application notebook dependencies
├── revision_simulations/              # Revision experiments, results, and reporting
└── README.md
```
> **Note:** The raw ADNI data (`ADNIMERGE_0.0.1.tar.gz`) and any individual-level ADNI CSV files are **not included** due to data use restrictions. 
> Where to Find and Access the Data: Alzheimer’s Disease Neuroimaging Initiative (ADNI)
>
> **Instructions to reproduce `ADNI_pc.csv`:**
> 1. Access process: Register for an account and submit a Data Use Application through the ADNI Image and Data Archive (IDA) Alzheimer's Journals.
> 2. Upon approval, you can download imaging and biomarker datasets in CSV or DICOM formats, including preprocessed PET measures (e.g., florbetapir SUVR by region) and MRI FreeSurfer outputs.
> 3. Go to tab Downloads --- Study Info --- Data&Database --- ADNIMERGE - Key ADNI tables merged into one table - Packages for R [ADNI1,GO,2]
> 4. PCA analysis on the selected features



> **Note:** The file `processed_mix3_df.rds` is derived from single-cell RNA-seq data (Zheng et al., 2017) and processed as in:
> Gao, L. L., Bien, J., & Witten, D. (2024). *Selective inference for hierarchical clustering.* Journal of the American Statistical Association, 119(545), 332–342.  
> Original RNA-seq data and R code are available at:  
> [https://github.com/lucylgao/clusterpval-experiments/blob/master/real-data-code/zheng.R](https://github.com/lucylgao/clusterpval-experiments/blob/master/real-data-code/zheng.R)
>
> **Instructions to reproduce `processed_mix3_df.rds`:**
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
   - Notebook: `FCM_weight.ipynb`
   - Description: Simulation study for (i) Consistency, (ii) Asymptotic Normality.
4. Application 
    - Example 1: RNA-seq (Single-Cell) Data
        - Notebook: `FCM_cell.ipynb`
        - Data: `data/processed_mix3_df.rds`
        - Description: Implements WFCM on a synthetic RNA-seq mixture dataset. Demonstrates clustering structure, soft memberships, and inference results.
    - Example 2: ADNI (Alzheimer's Disease) Data
        - Notebook: `FCM_adni.ipynb`
        - Data: `data/ADNI_test.csv` — revised input with PC1, PC2, PC3 and clinical labels
                `ADNI_data.csv` — extracted ADNI features before PCA
                `ADNIMERGE_0.0.1.tar.gz` — raw ADNI dataset
        - Description: Revised analysis from `FCM_adni_rev.ipynb`, retained under the existing repository filename. Includes the appended likelihood-ratio test and an optional pre-PCA Kaiser–Guttman diagnostic.
    - Example 3 (revision): TCGA-BRCA
        - Notebook: `TCGA_BRCA_three_methods.ipynb`
        - Compares K-means, FCM, and WFCM against PAM50 subtypes; exports fitting diagnostics, memberships, metrics, and supplementary plots.
        - Configure the expression and clinical inputs in the notebook settings. The expression matrix must have an `Ensembl_ID` column; the clinical table must have `sampleID` and `PAM50Call_RNAseq`. The notebook checks the expected 819-sample cohort before fitting.
        - Defaults copied from the supplied revision notebook: 5,000 variable genes, five PCs, four clusters, and fixed m=2. Prepared/existing input modes require the same five-PC representation.
        - Alternative embeddings visualize the fitted analysis; they do not refit the methods.


## Running the application notebooks

Install dependencies with `python -m pip install -r requirements-applications.txt`,
start Jupyter from the repository root, and run each notebook from top to bottom.
For optional TCGA UMAP panels, install `umap-learn` separately before execution.
Simulation dependencies and instructions remain in `revision_simulations/`.

Provide data locally in `data/` (not tracked). The single-cell notebook reads
`data/processed_mix3_df.rds`; the revised ADNI notebook reads `data/ADNI_test.csv`.
Keep the ADNI CSV's clinical columns used by the later plotting/evaluation cells.
TCGA expression and prepared-input paths are configurable in its settings cell.
The clinical source defaults to a public Xena URL and can be replaced by a local
saved table for an offline run. Raw TCGA expression values are used as supplied;
confirm their normalization/transform against the paper before substituting a matrix.

Individual-level datasets are not bundled. ADNI requires approved access, and the
single-cell processed input must be generated separately as described above.
Notebook outputs are cleared before publication to avoid distributing individual
records or implying that old outputs were generated by revised code. Full numerical
reproduction requires the original data and running the notebooks.
The supplied single-cell subsampling uses NumPy's global RNG without an explicit
seed before sampling; set and record a seed for a new run, or retain the original
sample indices to reproduce the manuscript cohort exactly.

## PCA checks added for the revision

The single-cell notebook retains the supplied variance-explained check. Single-cell
and raw-mode TCGA additionally report the Kaiser–Guttman count and eigenvalue table
using the correlation matrix of the complete pre-PCA feature set (eigenvalues >1).
These diagnostics do not alter the original PCA dimensions or clustering results.
Their standardized correlation PCA can differ from the main unscaled PCA.
See the [criterion documentation](https://search.r-project.org/CRAN/refmans/EFAtools/html/KGC.html).

ADNI starts with saved PCs, so its Kaiser–Guttman cell is optional until
`ADNI_PRE_PCA_PATH` and `ADNI_FEATURE_COLUMNS` are supplied. Use the same subjects,
QC, imputation, and original PCA features; exclude diagnoses and identifiers.
Applying the rule to only three standardized PCs would not establish a retention
criterion for the original feature space.

The appended ADNI likelihood-ratio analysis retains the supplied chi-square
reference with degrees of freedom equal to the PCA dimension. Its calibration,
including coincident-center non-identifiability and data-selected m, needs to be
justified by the paper's theory or null simulation/bootstrap. The notebook does
not validate that reference distribution and stops if the LR statistic is invalid.

## Citation
If you use this code in your research, please cite:

Wu, Qiuyi, Zihan Zhu, and Anru R. Zhang. "Statistical Inference for Fuzzy Clustering." arXiv preprint arXiv:2601.02656 (2026).

## Contact

For questions, please contact:
Qiuyi Wu (qiuyi.wu@duke.edu) or Zihan Zhu (zhzhu1@wharton.upenn.edu).


