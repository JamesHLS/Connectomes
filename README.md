# Structural and Functional Connectome Modelling

How well can the brain's structural wiring predict its functional connectivity? This project investigates that relationship through MRI data exploration, graph analysis, regression, and sparse predictive modelling.

The analysis combines single-subject MRI data with structural and functional connectivity matrices from 19 healthy adults. It examines how network construction choices affect graph properties, compares direct and indirect structural predictors, and evaluates prediction across subjects.

Methods, results, and discussion are included directly in the notebooks, alongside the code outputs they explain. Start with [MRI exploration](mri_data_exploration.ipynb), then follow the modelling and evaluation notebooks below.

## Analysis workflow

| Focus | Notebook | What it explores |
| --- | --- | --- |
| MRI exploration | [mri_data_exploration.ipynb](mri_data_exploration.ipynb) | Image volumes, anatomical labels, and FA thresholding |
| Structural networks | [structural_graph_analysis.ipynb](structural_graph_analysis.ipynb) | Network density, shortest paths, efficiency, and clustering across FA thresholds |
| Functional networks | [functional_connectome_shrinkage.ipynb](functional_connectome_shrinkage.ipynb) | Correlation shrinkage and the effect of retaining negative connections |
| Regression baselines | [connectivity_regression.ipynb](connectivity_regression.ipynb) | Five models using direct and indirect structural connections |
| Model evaluation | [model_comparison.ipynb](model_comparison.ipynb) | AIC/BIC, cross-validation, individual connections, and pooled models |
| Multistep connections | [multistep_connectivity_model.ipynb](multistep_connectivity_model.ipynb) | A predictor combining direct, two-step, and three-step connectivity |
| Regional associations | [connectivity_density_modelling.ipynb](connectivity_density_modelling.ipynb) | Relationships between structural and functional connectivity density |
| Generalisation | [cross_subject_generalisation.ipynb](cross_subject_generalisation.ipynb) | Transfer of subject-specific models and leave-one-subject-out prediction |
| Sparse prediction | [lasso_connectivity_prediction.ipynb](lasso_connectivity_prediction.ipynb) | LASSO models and structural feature selection |

Additional exploratory notebooks provide [matrix comparisons](connectivity_matrix_exploration.ipynb), [network visualisations](connectivity_network_visualisation.ipynb), and [structural-functional overlap analysis](structural_functional_overlap.ipynb).

## Methods and tools

- Graph analysis: density, clustering, shortest paths, and global efficiency.
- Statistical modelling: ordinary least squares, AIC/BIC, cross-validation, and multiple-comparison correction.
- Sparse modelling: LASSO regression with nested cross-validation and descriptive feature selection.
- Python libraries: NumPy, pandas, SciPy, statsmodels, scikit-learn, Matplotlib, seaborn, NetworkX, NiBabel, and Nilearn.

## Project layout

```text
data/
  mri/                       Single-subject MRI and FA-thresholded structural graphs
  connectomes/               Connectivity matrices for subjects 32-50
figures/
  functional_connectomes/    Exported functional network figures
results/                    LASSO held-out predictions, edge metrics, and summary
*.ipynb                     Analysis and exploration notebooks
requirements.txt            Tested dependency versions
scripts/run_notebooks.py    Execute notebooks in fresh kernels
scripts/lasso_evaluation.py Fold-local tuning and nested LASSO evaluation
tests/                      Evaluation and leakage checks
```

## Running the notebooks

Open Jupyter with this directory as the working directory. In a Python 3.13 environment, install the tested dependencies and launch JupyterLab:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

Data paths are relative to the project root; each notebook contains its own data-loading code. To regenerate all saved outputs:

```sh
python scripts/run_notebooks.py
```

To run one analysis, pass its filename, for example `python scripts/run_notebooks.py model_comparison.ipynb`. The LASSO analysis uses up to six worker processes and takes several minutes.

All 12 notebooks were executed successfully in fresh kernels on 9 October 2026. The discussion retains the original report wording where supported, with verified numerical corrections. The LASSO methods and results were updated after correcting its evaluation: scaling and penalty selection now occur inside the training folds, and both models predict the same held-out subjects. Feature-selection plots describe separate full-data refits. See the [LASSO notebook](lasso_connectivity_prediction.ipynb) and [LASSO results](results/lasso_summary.json).

Run the evaluation checks with `python -m unittest discover -s tests -v`.
