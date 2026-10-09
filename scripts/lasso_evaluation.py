"""Fold-local LASSO selection and nested leave-one-subject-out evaluation.

The fixed alpha grid is independent of the observations. LARS evaluates its
piecewise-linear LASSO path efficiently; coordinate descent is used if LARS
reports a numerical problem. Both solve the same LASSO objective.
"""

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lars_path, lasso_path
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler


ALPHAS = np.geomspace(1.0, 1e-5, 50)


def _scaled_path(X, y, alphas):
    """Fit preprocessing and a coefficient path using only the supplied rows."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 1 or not len(alphas) or np.any(alphas <= 0):
        raise ValueError('alphas must be a nonempty vector of positive values')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('X and y must be finite')
    scaler = StandardScaler().fit(X)
    variable = scaler.var_ > 0
    X_scaled = np.asfortranarray(scaler.transform(X)[:, variable])
    mean_y = y.mean()
    centered_y = y - mean_y
    fallback = False
    if not variable.any() or np.all(centered_y == 0):
        knots = np.array([0.0])
        coefficients = np.zeros((variable.sum(), 1))
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            knots, _, coefficients = lars_path(
                X_scaled, centered_y, method='lasso', alpha_min=float(alphas.min()),
                max_iter=1000,
            )
        fallback = (
            any(issubclass(w.category, ConvergenceWarning) for w in caught)
            or not np.isfinite(coefficients).all()
            or np.any(np.diff(knots) >= 0)
            or knots[-1] > alphas.min() * (1 + 1e-8)
        )
        if fallback:
            # Do not hide a failure of the fallback solver.
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                knots, coefficients, _ = lasso_path(
                    X_scaled, centered_y,
                    alphas=np.unique(alphas)[::-1], tol=1e-4, max_iter=100000,
                )
    return scaler, variable, mean_y, knots, coefficients, int(fallback)


def path_predictions(X_train, y_train, X_test, alphas=ALPHAS):
    """Predict at every candidate alpha, with a scaler fitted on X_train only."""
    scaler, variable, mean_y, knots, coefficients, fallbacks = _scaled_path(
        X_train, y_train, alphas
    )
    active = np.any(coefficients != 0, axis=1)
    transformed = scaler.transform(np.asarray(X_test, dtype=np.float64))[:, variable]
    at_knots = transformed[:, active] @ coefficients[active] + mean_y
    predictions = np.vstack([
        np.interp(alphas, knots[::-1], row[::-1]) for row in at_knots
    ])
    return predictions, fallbacks


def select_alpha(X, y, alphas=ALPHAS, n_splits=5):
    """Five-fold selection with a fresh scaler for every inner training fold."""
    alphas = np.sort(np.unique(np.asarray(alphas, dtype=float)))[::-1]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    fold_errors = []
    fallbacks = 0
    for train, validation in splitter.split(X):
        predictions, count = path_predictions(
            X[train], y[train], X[validation], alphas
        )
        fold_errors.append(np.mean((y[validation, None] - predictions) ** 2, axis=0))
        fallbacks += count
    # Descending alpha order chooses the stronger penalty on an exact tie.
    best = int(np.argmin(np.mean(fold_errors, axis=0)))
    return float(alphas[best]), fallbacks


def nested_lasso_predictions(X, y, alphas=ALPHAS, inner_splits=5):
    """Return one independent prediction and selected alpha for every subject."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    predictions = np.empty(len(y))
    selected_alphas = np.empty(len(y))
    fallbacks = 0
    for train, test in LeaveOneOut().split(X):
        alpha, count = select_alpha(X[train], y[train], alphas, inner_splits)
        fold_prediction, fit_count = path_predictions(X[train], y[train], X[test], [alpha])
        predictions[test] = fold_prediction[:, 0]
        selected_alphas[test] = alpha
        fallbacks += count + fit_count
    return predictions, selected_alphas, fallbacks


def fit_descriptive_model(X, y, alphas=ALPHAS, inner_splits=5):
    """Separate full-data refit for feature plots, never used for scoring."""
    alpha, fallbacks = select_alpha(X, y, alphas, inner_splits)
    scaler, variable, mean_y, knots, path, count = _scaled_path(X, y, [alpha])
    coefficients = np.zeros(X.shape[1])
    coefficients[variable] = [np.interp(alpha, knots[::-1], row[::-1]) for row in path]
    return alpha, coefficients, fallbacks + count


def direct_ols_predictions(x, y):
    """LOOCV OLS; constant training predictors reduce to a training-mean model."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    predictions = np.empty(len(y))
    for train, test in LeaveOneOut().split(x):
        if np.ptp(x[train]) == 0:
            predictions[test] = y[train].mean()
        else:
            design = np.column_stack([np.ones(len(train)), x[train]])
            intercept, slope = np.linalg.lstsq(design, y[train], rcond=None)[0]
            predictions[test] = intercept + slope * x[test]
    return predictions
