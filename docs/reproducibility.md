# Reproducibility and provenance

The notebooks originate from the project's April 2026 submission archive. Their names and organisation describe the analyses rather than assignment tasks. All 12 notebooks have now been executed in fresh Python kernels, and their saved outputs have been regenerated.

The report's introduction, methods, results, tables, figure captions, discussion, and conclusion are distributed across nine analysis notebooks. Original wording is retained, including original spelling and task references within that prose. Equations and tables were reformatted for notebook Markdown; numerical corrections are listed below. New verification notes are labelled separately. The report PDF is not part of the portfolio repository.

The original [project brief](background/project_brief.pdf) and [lecture notes](background/lecture_notes.pdf) are retained as background material. The supplied ZIP and report remain outside the repository in the parent workspace directory.

## Original filenames

| Original | Current |
| --- | --- |
| `explore_tractor.ipynb` | `mri_data_exploration.ipynb` |
| `plot_graphs.ipynb` | `structural_graph_analysis.ipynb` |
| `corshrink_connectome.ipynb` | `functional_connectome_shrinkage.ipynb` |
| `Task2.ipynb` | `connectivity_regression.ipynb` |
| `Task2_3.ipynb` | `model_comparison.ipynb` |
| `Task2_5.ipynb` | `multistep_connectivity_model.ipynb` |
| `advanced_task1.ipynb` | `connectivity_density_modelling.ipynb` |
| `advanced_task2.ipynb` | `cross_subject_generalisation.ipynb` |
| `advanced_task3.ipynb` | `lasso_connectivity_prediction.ipynb` |
| `WFA_vs_rsfMRI.ipynb` | `connectivity_matrix_exploration.ipynb` |
| `func_v_struc.ipynb` | `connectivity_network_visualisation.ipynb` |
| `fs_overlaps.ipynb` | `structural_functional_overlap.ipynb` |
| `Task1Data/` | `data/mri/` |
| `Task2Data/` | `data/connectomes/` |
| `Task1_outputs/` | `figures/functional_connectomes/` |

## Execution status

All notebooks completed without execution errors on 9 October 2026 using Python 3.13.5. [requirements.txt](../requirements.txt) records the dependency versions used. [execution_results.json](execution_results.json) records notebook timings, executed-cell counts, and code hashes. After execution, only Markdown and notebook metadata were changed to integrate the report; the regenerated code outputs were preserved.

Run [scripts/run_notebooks.py](../scripts/run_notebooks.py) from the project environment to repeat the analyses. Each notebook uses a fresh kernel. The LASSO fits are parallelised across independent connections with unchanged model settings and validation procedure.

Execution repairs included:

- Defining the example connection before its use in the regression notebook.
- Reading the matrix CSV metadata correctly and completing the unfinished label assignment in matrix exploration.
- Keeping separate layouts for the full network and its intersection in the overlap visualisation.
- Treating non-finite background voxels as zero when displaying and thresholding the FA image.
- Updating data paths, directing new image exports into `figures/`, and making structural-graph filenames portable.
- Removing package-install commands from notebook execution, upgrading notebook schemas to support cell IDs, and correcting an unreadable plot legend.

## Report placement

| Notebook | Report content |
| --- | --- |
| MRI data exploration | Introduction, dataset description, Figure 1, spatial FA-threshold discussion |
| Structural graph analysis | Structural thresholding methods, Table 1, graph-metric discussion |
| Functional connectome shrinkage | Shrinkage methods, Table 2, negative-edge and shrinkage discussion |
| Connectivity regression | Direct and indirect connectivity definitions |
| Model comparison | Evaluation methods and interpretation of direct/indirect predictors |
| Multistep connectivity model | Six regression equations, Table 3, Figures 2–4, model and scatter-plot discussion |
| Connectivity density modelling | Regional-density methods, Table 4, significance and slope discussion |
| Cross-subject generalisation | Transfer methods, Table 5, global-fit and cross-subject discussion |
| LASSO connectivity prediction | LASSO methods, Table 6, Figures 5–6, feature-selection discussion and project conclusion |

Figure and table numbering follows the original report. The three supplementary exploration notebooks were also run successfully; the report did not contain dedicated discussion sections for them.

## Verified numerical corrections

| Location | Original value | Value from execution |
| --- | --- | --- |
| Shrinkage Table 2, negative edges discarded, lambda = 0.2, clustering | 0.683 | 0.682 |
| Density discussion, superior frontal (L) slope | +0.061 | −0.061 |
| Density discussion, negative / positive slope counts | 35 / 33 | 42 / 26 |
| Cross-subject discussion, subject 35's MSE across five models | 0.009044 | 0.008954–0.009710 (model 1 remains 0.009044) |
| LASSO discussion, edges with a direct structural connection in at least one subject | 1422 | 1522 |

The other reported rounded model-comparison and LASSO summary statistics were reproduced. These corrections change numeric values or signs, not the surrounding report wording.

## Interpretation and validation notes

Some original statements cannot be corrected by replacing numbers alone; they are retained with separate verification notes beside the relevant results:

- The supplied multi-subject functional matrices use normalised precision matrices, while the single-subject analysis uses shrinkage correlations.
- The quadratic models have higher, not lower, mean AIC than their linear counterparts. Comparisons using different valid-edge sets also require care.
- Approximately 7% explained variance describes the subject-specific global models; it is not a reported average of the edge-specific regressions. Similar training and cross-subject errors alone do not demonstrate strong predictive accuracy.
- The original LASSO protocol scales features and selects alpha before the outer leave-one-out predictions. It is not nested validation and can produce optimistic errors. The no-direct-connection OLS baseline uses full-sample variance rather than leave-one-out mean predictions. These procedures were preserved to reproduce the original analysis.
