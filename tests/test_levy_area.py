"""Order-2 log-signature: geometric properties and vectorised/scalar equivalence."""

import numpy as np
import pytest

from stillwar.features import compute_increments_and_levy_areas, logsig_features


def test_straight_line_zero_levy_area() -> None:
    """All ten Levy areas vanish identically along a collinear path."""
    grid = np.linspace(0.0, 1.0, 11)
    path = np.column_stack([grid * 1.0, grid * 2.0, grid * -0.5, grid * 3.0, grid * 0.1])
    increments, levy = compute_increments_and_levy_areas(path)

    assert increments.shape == (5,)
    assert levy.shape == (10,)
    np.testing.assert_allclose(increments, path[-1] - path[0], atol=1e-12)
    np.testing.assert_allclose(levy, np.zeros(10), atol=1e-12)


def test_geometric_rectangle_signed_area() -> None:
    """A closed counter-clockwise rectangle of size ``a x b`` yields Levy area ``+a * b``."""
    width, height = 2.0, 3.0
    rect = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [width, 0.0, 0.0, 0.0, 0.0],
            [width, height, 0.0, 0.0, 0.0],
            [0.0, height, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    increments, levy = compute_increments_and_levy_areas(rect)

    np.testing.assert_allclose(increments, np.zeros(5), atol=1e-12)
    assert levy[0] == pytest.approx(width * height, abs=1e-12)
    np.testing.assert_allclose(levy[1:], np.zeros(9), atol=1e-12)


def test_scaling_property() -> None:
    """Scaling path coordinates by ``c`` scales Level-2 Levy areas by ``c**2``."""
    rng = np.random.default_rng(42)
    path = rng.standard_normal((21, 5))
    _, levy_orig = compute_increments_and_levy_areas(path)
    _, levy_scaled = compute_increments_and_levy_areas(2.5 * path)
    np.testing.assert_allclose(levy_scaled, 6.25 * levy_orig, atol=1e-12)


@pytest.mark.parametrize("windows", [(5,), (5, 20), (5, 20, 60)])
def test_vectorised_matches_scalar_reference(windows: tuple[int, ...]) -> None:
    """``logsig_features`` reproduces the scalar reference on every window and asset.

    This is the equivalence test the repository previously lacked: the pipeline only
    ever calls the vectorised path, and the geometric tests only exercised the scalar
    one, so a defect in the cumulative-sum identity would have gone unnoticed.
    """
    rng = np.random.default_rng(2026)
    t_len, n_assets = 90, 3
    path = rng.standard_normal((t_len, n_assets, 5))

    vectorised = logsig_features(path, windows)

    for t_idx in (60, 71, 89):
        for asset in range(n_assets):
            expected = []
            for window in windows:
                sub = path[t_idx - window : t_idx + 1, asset]
                increments, levy = compute_increments_and_levy_areas(sub)
                expected.append(np.concatenate([increments, levy]))
            np.testing.assert_allclose(
                vectorised[t_idx, asset], np.concatenate(expected), atol=1e-10
            )


def test_warmup_rows_are_nan() -> None:
    """Each window's block is NaN until that window has enough history."""
    rng = np.random.default_rng(1)
    out = logsig_features(rng.standard_normal((80, 2, 5)), (5, 20, 60))
    assert out.shape == (80, 2, 45)

    for block, window in enumerate((5, 20, 60)):
        chunk = out[:, :, block * 15 : (block + 1) * 15]
        assert np.isnan(chunk[:window]).all(), f"window {window} leaks before its warm-up"
        assert np.isfinite(chunk[window:]).all()


def test_translation_invariance() -> None:
    """Adding a per-channel constant leaves increments and areas unchanged."""
    rng = np.random.default_rng(3)
    path = rng.standard_normal((80, 2, 5))
    shifted = path + np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    np.testing.assert_allclose(
        logsig_features(path, (5, 20, 60)), logsig_features(shifted, (5, 20, 60)), atol=1e-10
    )
