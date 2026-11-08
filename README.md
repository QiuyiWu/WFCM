# Statistical Inference for Fuzzy Clustering (WFCM)

This repository contains the analysis code accompanying *Statistical Inference for Fuzzy Clustering* by Qiuyi Wu, Zihan Zhu, and Anru R. Zhang.

- [FCM_figure.ipynb](FCM_figure.ipynb) generates the method illustrations.
- [FCM_weight.ipynb](FCM_weight.ipynb) contains the original simulation notebook.
- [revision_simulations/](revision_simulations/) contains the revision simulation code, saved simulation results, and reporting commands.

## Application data and setup

The three application notebooks require separate data inputs. No application datasets are bundled. Keep downloads and individual-level exports local. Place the inputs in a local `data/` directory or set the `WFCM_DATA_DIR` environment variable to the directory containing them before starting Jupyter. Use a fresh kernel for each notebook and run the cells in order.

Install the common Python dependencies in the environment you use for Jupyter.

```sh
python -m pip install jupyter numpy pandas scipy scikit-learn matplotlib seaborn pyreadr pillow
```

The single-cell notebook reads RDS files with `pyreadr`. Install `umap-learn` separately to reproduce the UMAP panels in the breast-cancer notebook.

### Single-cell RNA-seq

**Notebook** [FCM_cell.ipynb](FCM_cell.ipynb)

**Local input** `data/processed_mix3_df.rds`

1. Download the filtered 10x count matrices for the Zheng et al. (2017) memory T cells, B cells, and monocytes using the data links in the [Gao, Bien, and Witten reproduction repository](https://github.com/lucylgao/clusterpval-experiments#real-data-code).
2. Follow that repository's directory setup. Under `real-data-code/raw/`, place the three extracted folders as `filtered_matrices_mex_memory`, `filtered_matrices_mex_bcell`, and `filtered_matrices_mex_mono`. Each folder should contain its `hg19` matrix directory.
3. From `real-data-code/`, run the preprocessing part of `zheng.R` through `processed.mix3 <- process_data(mix3)`. Then export the processed mixture as a data frame with one row per cell, numeric gene-expression columns, and a `CellType` column. [Upstream preprocessing script](https://github.com/lucylgao/clusterpval-experiments/blob/master/real-data-code/zheng.R)

   ```r
   df <- as.data.frame(processed.mix3$data)
   df$CellType <- processed.mix3$labels
   saveRDS(df, "processed_mix3_df.rds")
   ```

4. Move the RDS to your local data directory. The notebook samples 1,500 memory T cells, 1,000 B cells, and 500 monocytes, then computes and standardizes the two retained PC scores.

### Alzheimer's Disease Neuroimaging Initiative (ADNI)

**Notebook** [FCM_adni.ipynb](FCM_adni.ipynb)

**Local input** `data/ADNI_data.csv` in the default `INPUT_MODE = 'raw'`

1. Apply for ADNI data access and accept the Data Use Agreement. After approval, sign in to the LONI Image and Data Archive (IDA), open **Study Files**, and look under **Study Info** for the merged **ADNIMERGE** table. [ADNI data access](https://adni.loni.usc.edu/data-samples/adni-data/) · [ADNI download FAQ](https://adni.loni.usc.edu/help-faqs/faqs/)
2. Prepare a baseline-only table with one row per participant. Use baseline records (`VISCODE = bl`) and retain the earliest baseline record if a participant has more than one. Export the following columns as `ADNI_data.csv`. Preserve missing values so the notebook can apply its own imputation and diagnosis handling.

   ```text
   AGE, PTEDUCAT, APOE4, CDRSB, ADAS11, ADAS13, ADASQ4, MMSE,
   RAVLT.perc.forgetting, LDELTOTAL, TRABSCOR, FAQ, DX
   ```

3. Run the notebook to impute and standardize the 12 predictors, compute PCA, and retain and re-standardize the first three PC scores for WFCM. The study input contains 2,436 participants. Exact reproduction also requires the study's ADNIMERGE release, cohort, and row order. An export from a different release may yield different results.

The notebook writes `ADNI_pc_final.csv` locally. If you already have the exact reference score file, you can place it in your data directory and choose `INPUT_MODE = 'corrected_scores'`. This mode checks the reference score fingerprint. Keep the ADNIMERGE files, participant-level CSVs, and outputs derived from individual records within your approved ADNI environment.

### TCGA-BRCA breast-cancer subtypes

**Notebook** [FCM_breast.ipynb](FCM_breast.ipynb)

**Local inputs for raw mode** `data/TCGA-BRCA.star_tpm.tsv.gz` and the matching Xena clinical table

1. Download the STAR TPM expression matrix from the UCSC Xena GDC hub. Keep the compressed filename `TCGA-BRCA.star_tpm.tsv.gz`. The notebook expects an `Ensembl_ID` column followed by TCGA sample columns. [Download the expression matrix](https://gdc.xenahubs.net/download/TCGA-BRCA.star_tpm.tsv.gz)
2. Use the UCSC Xena TCGA-BRCA clinical matrix with columns `sampleID` and `PAM50Call_RNAseq`. The notebook defaults to the Xena URL in `CLINICAL_SOURCE`. For an offline run, download the clinical matrix and set `CLINICAL_SOURCE` to its local path. [Download the clinical matrix](https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix)
3. Set `INPUT_MODE = 'raw'` in the settings cell. The notebook retains one primary tumor per participant, matches PAM50 annotations at the sample level, selects the 5,000 most variable genes, and computes five PCs from the supplied expression values. The reference cohort contains 819 tumors (139 Basal, 67 Her2, 421 LumA, and 192 LumB). The run exports `results_breast/prepared_pcs.csv` locally.

If you already have the exact reference PC scores, the default `INPUT_MODE = 'prepared'` reads `data/TCGA_BRCA_prepared_pcs.csv`. The first CSV column must be a unique sample ID, followed by `PC1` through `PC5` and `PAM50`. This mode checks the reference fingerprints for the scores, sample IDs, and labels.

To run the optional 3-, 5-, 10-, and 20-PC sensitivity sweep, set `RUN_PC_SENSITIVITY = True` in the settings cell. The sweep needs the expression matrix and clinical table in either input mode. The default setting runs only the main five-PC analysis.

## Citation
If you use this code in your research, please cite:

Wu, Qiuyi, Zihan Zhu, and Anru R. Zhang. "Statistical Inference for Fuzzy Clustering." arXiv preprint arXiv:2601.02656 (2026).

## Contact

For questions, please contact:
Qiuyi Wu (qiuyi.wu@duke.edu) or Zihan Zhu (zhzhu1@wharton.upenn.edu).
