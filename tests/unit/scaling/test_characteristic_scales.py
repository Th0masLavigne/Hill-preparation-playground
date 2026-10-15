"""Unit tests of ``CharacteristicScales`` (copied from fenicsoft's demo-side ``test/``, import path updated)."""

import pytest

from hill_playground.scaling import CharacteristicScales


def test_construction_stores_every_field() -> None:
	"""Confirm the four base scales are stored as given."""
	scales = CharacteristicScales(length=1.5e-3, stress=2.0e3, time=0.5, density=1.0e3)
	assert scales.length == 1.5e-3
	assert scales.stress == 2.0e3
	assert scales.time == 0.5
	assert scales.density == 1.0e3


def test_is_frozen() -> None:
	"""CharacteristicScales is immutable: every field cannot be reassigned after construction."""
	scales = CharacteristicScales(length=1.0, stress=1.0, time=1.0, density=1.0)
	with pytest.raises(AttributeError):
		scales.length = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["length", "stress", "time", "density"])
def test_rejects_zero_or_negative_base_scale(field_name: str) -> None:
	"""Each base scale must be strictly positive: zero or negative must raise ValueError."""
	kwargs = {"length": 1.0, "stress": 1.0, "time": 1.0, "density": 1.0}
	kwargs[field_name] = 0.0
	with pytest.raises(ValueError, match=field_name):
		CharacteristicScales(**kwargs)

	kwargs[field_name] = -1.0
	with pytest.raises(ValueError, match=field_name):
		CharacteristicScales(**kwargs)


def test_from_consolidation_scale_matches_the_derivation_formula() -> None:
	"""Time = length**2 * viscosity / (permeability * consolidation_modulus), per the module docstring.

	Uses a ``consolidation_modulus`` deliberately different from ``stress`` to confirm the two
	are not conflated internally (a regression test for the mu0-vs-kappa_gel distinction -- see
	the module docstring's own "Why four base scales" section).
	"""
	length, stress, density = 1.5e-3, 2.0e3, 1.0e3
	permeability, viscosity, consolidation_modulus = 1.0e-14, 1.0e-3, 1.7e3

	scales = CharacteristicScales.from_consolidation_scale(
		length=length,
		stress=stress,
		density=density,
		permeability=permeability,
		viscosity=viscosity,
		consolidation_modulus=consolidation_modulus,
	)

	expected_time = length**2 * viscosity / (permeability * consolidation_modulus)
	assert scales.time == pytest.approx(expected_time, rel=1e-12)
	assert scales.time != pytest.approx(length**2 * viscosity / (permeability * stress), rel=1e-6)
	assert scales.length == length
	assert scales.stress == stress
	assert scales.density == density


@pytest.mark.parametrize("bad_field", ["permeability", "viscosity", "consolidation_modulus"])
def test_from_consolidation_scale_rejects_non_positive_gel_parameters(bad_field: str) -> None:
	"""A non-positive permeability, viscosity, or consolidation_modulus is unphysical and must raise ValueError."""
	kwargs = {
		"length": 1.0e-3,
		"stress": 1.0e3,
		"density": 1.0e3,
		"permeability": 1.0e-14,
		"viscosity": 1.0e-3,
		"consolidation_modulus": 8.0e2,
	}
	kwargs[bad_field] = 0.0
	with pytest.raises(ValueError, match=bad_field):
		CharacteristicScales.from_consolidation_scale(**kwargs)


def test_derived_scales_formulas() -> None:
	"""velocity, diffusivity, rate and permeability_over_viscosity follow their own formulas."""
	scales = CharacteristicScales(length=2.0, stress=4.0, time=5.0, density=1.0)
	assert scales.velocity == pytest.approx(2.0 / 5.0)
	assert scales.diffusivity == pytest.approx(2.0**2 / 5.0)
	assert scales.rate == pytest.approx(1.0 / 5.0)
	assert scales.permeability_over_viscosity == pytest.approx(2.0**2 / (4.0 * 5.0))


def test_from_consolidation_scale_makes_the_darcy_block_o1_when_stress_equals_kappa() -> None:
	"""permeability_over_viscosity == permeability/viscosity exactly when stress == consolidation_modulus.

	This is the O(1)-Darcy-block property the time scale is derived to guarantee (see the
	module docstring): with ``time`` built this way, and ``consolidation_modulus`` equal to
	``stress``, the characteristic hydraulic conductivity exactly reproduces the gel's own
	k/eta, so DarcyGradient's nondimensionalized conductivity argument is exactly 1.0.
	"""
	permeability, viscosity, modulus = 1.0e-14, 1.0e-3, 2.0e3
	scales = CharacteristicScales.from_consolidation_scale(
		length=1.5e-3,
		stress=modulus,
		density=1.0e3,
		permeability=permeability,
		viscosity=viscosity,
		consolidation_modulus=modulus,
	)
	assert scales.permeability_over_viscosity == pytest.approx(permeability / viscosity, rel=1e-12)


def test_from_consolidation_scale_scales_permeability_over_viscosity_by_stress_over_kappa() -> None:
	"""When stress != consolidation_modulus: scale = (k/eta) kappa/mu0, hence K_hat = mu0/kappa.

	L0^2/(mu0 t0) with t0 = L0^2 eta/(k kappa) gives (k/eta) kappa/mu0 (equations.tex, Table
	tab:groups: K_hat_gel = mu0/kappa_gel = 1.20). Fixed from the fenicsoft demo-side original,
	which expected the inverted ratio mu0/kappa for the scale itself.
	"""
	permeability, viscosity, kappa_gel, mu0 = 1.0e-14, 1.0e-3, 1417.0, 1700.0
	scales = CharacteristicScales.from_consolidation_scale(
		length=1.5e-3,
		stress=mu0,
		density=1.0e3,
		permeability=permeability,
		viscosity=viscosity,
		consolidation_modulus=kappa_gel,
	)
	expected = (permeability / viscosity) * (kappa_gel / mu0)
	assert scales.permeability_over_viscosity == pytest.approx(expected, rel=1e-12)
	k_hat = scales.nondimensionalize_permeability_over_viscosity(permeability / viscosity)
	assert k_hat == pytest.approx(mu0 / kappa_gel, rel=1e-12)


@pytest.mark.parametrize(
	"quantity_name,raw_value",
	[
		("length", 3.7e-4),
		("stress", 812.5),
		("time", 0.042),
		("rate", 6.0),
		("diffusivity", 1.1e-9),
		("permeability_over_viscosity", 5.5e-12),
		("density", 950.0),
	],
)
def test_nondimensionalize_redimensionalize_round_trip(quantity_name: str, raw_value: float) -> None:
	"""redimensionalize_X(nondimensionalize_X(value)) == value for every conversion pair.

	This is the mathematical property that makes solving the rescaled problem and exporting
	its redimensionalized solution equivalent to solving the original SI problem directly (see
	the module docstring's discussion of R = N * R_hat).
	"""
	scales = CharacteristicScales.from_consolidation_scale(
		length=1.5e-3,
		stress=2.0e3,
		density=1.0e3,
		permeability=1.0e-14,
		viscosity=1.0e-3,
		consolidation_modulus=1.7e3,
	)
	nondim = getattr(scales, f"nondimensionalize_{quantity_name}")
	redim = getattr(scales, f"redimensionalize_{quantity_name}")

	assert redim(nondim(raw_value)) == pytest.approx(raw_value, rel=1e-12)
	assert nondim(redim(raw_value)) == pytest.approx(raw_value, rel=1e-12)
