import unittest
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

from scripts.lasso_evaluation import (
    direct_ols_predictions, nested_lasso_predictions, path_predictions, select_alpha,
)


class NestedEvaluationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(12, 5))
        self.y = self.X[:, 0] * 0.2 + rng.normal(scale=0.08, size=12)
        self.alphas = np.array([0.2, 0.06, 0.02, 0.006])

    def test_zero_edge_baseline_uses_training_mean(self):
        y = np.array([1., 3., 8., 20.])
        expected = (y.sum() - y) / (len(y) - 1)
        np.testing.assert_allclose(direct_ols_predictions(np.zeros(4), y), expected)

    def test_training_constant_even_if_test_edge_exists(self):
        x = np.array([8., 0., 0., 0.])
        y = np.array([100., 2., 3., 4.])
        self.assertEqual(direct_ols_predictions(x, y)[0], 3.)

    def test_held_out_extremes_do_not_change_tuning(self):
        _, alphas, _ = nested_lasso_predictions(self.X, self.y, self.alphas, 3)
        changed_x = self.X.copy(); changed_y = self.y.copy()
        changed_x[0] = 1e6; changed_y[0] = -1e6
        _, changed_alphas, _ = nested_lasso_predictions(changed_x, changed_y, self.alphas, 3)
        self.assertEqual(alphas[0], changed_alphas[0])

    def test_every_inner_scaler_sees_only_training_rows(self):
        seen = []
        original_fit = StandardScaler.fit
        def recording_fit(scaler, X, *args, **kwargs):
            seen.append(X.copy())
            return original_fit(scaler, X, *args, **kwargs)
        with patch.object(StandardScaler, 'fit', recording_fit):
            select_alpha(self.X, self.y, self.alphas, 3)
        expected = [self.X[train] for train, _ in KFold(3, shuffle=True, random_state=0).split(self.X)]
        self.assertEqual(len(seen), 3)
        for actual, wanted in zip(seen, expected):
            np.testing.assert_array_equal(actual, wanted)

    def test_nested_predictions_match_pipeline_grid_search(self):
        actual, selected, _ = nested_lasso_predictions(self.X, self.y, self.alphas, 3)
        expected = np.empty(len(self.y))
        for train, test in LeaveOneOut().split(self.X):
            search = GridSearchCV(
                make_pipeline(StandardScaler(), Lasso(tol=1e-10, max_iter=100000)),
                {'lasso__alpha': self.alphas},
                cv=KFold(3, shuffle=True, random_state=0), scoring='neg_mean_squared_error',
            ).fit(self.X[train], self.y[train])
            expected[test] = search.predict(self.X[test])
            self.assertEqual(selected[test][0], search.best_params_['lasso__alpha'])
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_constant_feature_only_in_test_cannot_affect_prediction(self):
        X = np.column_stack([self.X, np.zeros(len(self.X))])
        extreme = X[-2:].copy(); extreme[:, -1] = 1e9
        actual, _ = path_predictions(X[:-2], self.y[:-2], extreme, self.alphas)
        expected, _ = path_predictions(X[:-2], self.y[:-2], X[-2:], self.alphas)
        np.testing.assert_allclose(actual, expected)

    def test_held_out_target_cannot_change_its_own_prediction(self):
        predicted, alphas, _ = nested_lasso_predictions(self.X, self.y, self.alphas, 3)
        changed_y = self.y.copy()
        changed_y[0] += 1000
        changed, changed_alphas, _ = nested_lasso_predictions(self.X, changed_y, self.alphas, 3)
        self.assertEqual(alphas[0], changed_alphas[0])
        self.assertAlmostEqual(predicted[0], changed[0], places=12)

    def test_degenerate_path_falls_back_to_coordinate_descent(self):
        def broken_path(X, y, **kwargs):
            warnings.warn('Numerically degenerate path', ConvergenceWarning)
            return np.array([1.]), [], np.zeros((X.shape[1], 1))
        with patch('scripts.lasso_evaluation.lars_path', side_effect=broken_path):
            actual, fallback = path_predictions(self.X[:-2], self.y[:-2], self.X[-2:], self.alphas)
        self.assertEqual(fallback, 1)
        expected = np.column_stack([
            make_pipeline(StandardScaler(), Lasso(alpha=a, tol=1e-10, max_iter=100000))
            .fit(self.X[:-2], self.y[:-2]).predict(self.X[-2:])
            for a in self.alphas
        ])
        # The fallback uses sklearn's standard 1e-4 optimisation tolerance;
        # compare with a tightly converged reference to prediction precision.
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=0)


if __name__ == '__main__':
    unittest.main()
