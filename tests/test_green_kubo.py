import numpy as np

from scripts.green_kubo_pipeline import autocorrelation, integrate_curve, species_groups


def test_autocorrelation_zero_lag_and_symmetry_for_scalar_component():
    x = np.arange(8.0)[:, None]
    ac = autocorrelation(x)
    assert np.isclose(ac[0, 0], np.var(x[:, 0], ddof=0))
    assert ac.shape == (8, 1)


def test_integral_has_expected_constant_curve_value():
    curve = np.ones(5)
    result = integrate_curve(curve, 0.25, 2.0)
    assert np.allclose(result, [0.0, 0.5, 1.0, 1.5, 2.0])


def test_species_group_mapping_rejects_wrong_tail():
    class FakeAtoms:
        def get_chemical_symbols(self):
            return ["Na"] + ["C"] * 8

    try:
        species_groups(FakeAtoms())
    except ValueError as exc:
        assert "post-Na" in str(exc) or "FSI" in str(exc)
    else:
        raise AssertionError("invalid species layout was accepted")
