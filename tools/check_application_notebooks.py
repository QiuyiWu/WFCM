"""Validate release notebooks and the added PCA diagnostic without private data.

Run from the repository root: python tools/check_application_notebooks.py
This does not execute the real-data analyses or validate LRT calibration.
"""
import ast
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['FCM_cell.ipynb', 'FCM_adni.ipynb', 'TCGA_BRCA_three_methods.ipynb']
rng = np.random.default_rng(127)
x = rng.normal(size=(40, 6))
x[:, 1] = 0.9 * x[:, 0] + 0.1 * x[:, 1]
# Constants are intentionally excluded from correlation PCA.
x = np.column_stack([x, np.ones(len(x))])
expected = np.linalg.eigvalsh(np.corrcoef(x[:, :-1], rowvar=False))[::-1]
for name in NAMES:
    notebook = nbformat.read(ROOT / name, as_version=4)
    nbformat.validate(notebook)
    definitions = []
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        assert not cell.outputs and cell.execution_count is None
        definitions.extend(node for node in tree.body
                           if isinstance(node, ast.FunctionDef) and node.name == 'kaiser_guttman')
    assert len(definitions) == 1
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), name, 'exec'), namespace)
    diagnostic = namespace['kaiser_guttman']
    table = diagnostic(x)
    np.testing.assert_allclose(table.correlation_eigenvalue, expected, atol=1e-12)
    np.testing.assert_allclose(table.variance_fraction.sum(), 1)
    # Measurement units must not affect the correlation-based diagnostic.
    scaled = diagnostic(x * np.arange(1, x.shape[1] + 1))
    np.testing.assert_allclose(scaled.correlation_eigenvalue, expected, atol=1e-12)
    wide = rng.normal(size=(5, 9))
    result = diagnostic(wide)
    np.testing.assert_allclose(result.correlation_eigenvalue.sum(), 9, atol=1e-12)
    assert len(result) == 9
    for bad in [np.ones((4, 3)), [[1, float('nan')], [2, 3]]]:
        try:
            diagnostic(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
    print(f'{name}: format, syntax, clean outputs, and Kaiser–Guttman checks passed')
