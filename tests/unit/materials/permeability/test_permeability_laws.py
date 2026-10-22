"""Unit tests of the permeability laws k/k0 and of K_L = (k/eta) J C^-1 on a homogeneous state F = j^(1/3) I."""

import math
from typing import Any, Callable

import dolfin
import pytest
from dolfin_mech.kinematics import Kinematics

from hill_playground.materials.permeability import base, compaction_power, constant, holmes_mow, kozeny_carman

PHI_S0, PHI_C0, K0 = 0.25, 0.05, 2.0


def _laws(kin: Kinematics, phi_c: float = PHI_C0) -> dict[str, Any]:
	common = dict(conductivity_0=K0, phi_s0=PHI_S0, phi_c=dolfin.Constant(phi_c), phi_c0=PHI_C0)
	return {
		"kozeny_carman": kozeny_carman.Permeability(kin, kozeny_carman.Parameters(**common)),
		"holmes_mow": holmes_mow.Permeability(kin, holmes_mow.Parameters(**common, exponent=2.0, stiffening=0.5)),
		"compaction_power": compaction_power.Permeability(kin, compaction_power.Parameters(**common, exponent=1.5)),
	}


@pytest.mark.parametrize("name", ["kozeny_carman", "holmes_mow", "compaction_power"])
def test_reference_state_gives_k0(
	name: str, mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	"""J = 1, Phi_c = Phi_c0: k = k0."""
	kin, mesh = dilate(1.0)
	assert mean(_laws(kin)[name].hydraulic_conductivity, mesh) == pytest.approx(K0, rel=1e-12)


@pytest.mark.parametrize("name", ["kozeny_carman", "holmes_mow", "compaction_power"])
def test_k_decreases_with_compaction_and_vanishes_at_jc(
	name: str, mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	"""k(J=0.8) < k0 and k -> 0 (floor) at J = J_min = Phi_s0 + Phi_c."""
	kin, mesh = dilate(0.8)
	assert mean(_laws(kin)[name].ratio, mesh) < 1.0
	kin, mesh = dilate(PHI_S0 + PHI_C0)
	assert mean(_laws(kin)[name].ratio, mesh) < 1e-6


@pytest.mark.parametrize("name", ["kozeny_carman", "holmes_mow", "compaction_power"])
def test_k_decreases_when_cells_proliferate(
	name: str, mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	"""Phi_c up at fixed J: k down."""
	kin, mesh = dilate(1.0)
	assert mean(_laws(kin, phi_c=0.1)[name].ratio, mesh) < 1.0


def test_closed_forms(mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]) -> None:
	"""Values at J = 0.8 against the formulas of the module docstrings."""
	jac = 0.8
	kin, mesh = dilate(jac)
	laws = _laws(kin)
	phi, phi_0 = 1.0 - (PHI_S0 + PHI_C0) / jac, 1.0 - PHI_S0 - PHI_C0
	rel = (jac - PHI_S0 - PHI_C0) / phi_0
	expected = {
		"kozeny_carman": (phi**3 / (1 - phi) ** 2) / (phi_0**3 / (1 - phi_0) ** 2),
		"holmes_mow": rel**2.0 * math.exp(0.25 * (jac**2 - 1.0)),
		"compaction_power": rel**1.5,
	}
	for name, value in expected.items():
		assert mean(laws[name].ratio, mesh) == pytest.approx(value, rel=1e-10), name


def test_tensor_is_pulled_back(mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]) -> None:
	"""K_L = (k/eta) J C^-1; constant law: K_L = k0 J C^-1."""
	jac = 0.8
	kin, mesh = dilate(jac)
	law = constant.Permeability(kin, constant.Parameters(conductivity_0=K0))
	assert mean(law.tensor[0, 0], mesh) == pytest.approx(K0 * jac / jac ** (2.0 / 3.0), rel=1e-10)
	kc = _laws(kin)["kozeny_carman"]
	assert mean(kc.tensor[1, 1] - kc.hydraulic_conductivity * jac ** (1.0 / 3.0), mesh) == pytest.approx(0.0, abs=1e-12)


def test_compaction_power_rejects_sublinear_exponent() -> None:
	"""Exponent n < 1 reaches J_min in finite time: refused."""
	with pytest.raises(ValueError):
		compaction_power.Parameters(conductivity_0=K0, phi_s0=PHI_S0, phi_c=PHI_C0, phi_c0=PHI_C0, exponent=0.5)


def test_parameters_and_porosity_accessors(
	mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	"""Every law exposes its parameters; phi_l = 1 - J_min/J."""
	jac = 0.8
	kin, mesh = dilate(jac)
	laws = _laws(kin)
	const = constant.Permeability(kin, constant.Parameters(conductivity_0=K0))
	assert const.parameters.conductivity_0 == K0
	assert const.ratio == 1.0
	for law in laws.values():
		assert law.parameters.phi_c0 == PHI_C0
		assert mean(law.porosity, mesh) == pytest.approx(1.0 - (PHI_S0 + PHI_C0) / jac, rel=1e-10)
		expected = mean(law.hydraulic_conductivity, mesh) * jac ** (1.0 / 3.0)  # (k/eta) J C^-1, F = J^(1/3) I
		assert mean(law.tensor[0, 0], mesh) == pytest.approx(expected, rel=1e-10)


def test_base_is_abstract_and_shares_porosity(
	mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	"""The base class needs a ratio; porous laws share phi_l0 and Phi_l/Phi_l0."""
	kin, mesh = dilate(0.8)
	with pytest.raises(TypeError):
		base.Permeability(kin, constant.Parameters(conductivity_0=K0))  # pyright: ignore[reportAbstractUsage]
	for law in _laws(kin).values():
		assert isinstance(law, base.PorousPermeability)
		assert law.porosity_0 == pytest.approx(1.0 - PHI_S0 - PHI_C0, rel=1e-12)
		assert mean(law.fluid_ratio, mesh) == pytest.approx(
			(0.8 - PHI_S0 - PHI_C0) / (1.0 - PHI_S0 - PHI_C0), rel=1e-12
		)


def test_uniform_solid_fraction_as_constant(
	mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	"""Phi_s0 given as a dolfin.Constant gives the same k."""
	kin, mesh = dilate(0.8)
	params = kozeny_carman.Parameters(
		conductivity_0=K0, phi_s0=dolfin.Constant(PHI_S0), phi_c=dolfin.Constant(PHI_C0), phi_c0=PHI_C0
	)
	law = kozeny_carman.Permeability(kin, params)
	assert mean(law.ratio, mesh) == pytest.approx(mean(_laws(kin)["kozeny_carman"].ratio, mesh), rel=1e-12)


@pytest.mark.parametrize("name", ["kozeny_carman", "holmes_mow", "compaction_power"])
def test_two_phase_mixture_is_phi_c_zero(
	name: str, mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	"""phi_c, phi_c0 omitted (solid + liquid) = the three-phase law with Phi_c = Phi_c0 = 0."""
	kin, mesh = dilate(0.8)
	module = {"kozeny_carman": kozeny_carman, "holmes_mow": holmes_mow, "compaction_power": compaction_power}[name]
	two_phase = module.Permeability(kin, module.Parameters(conductivity_0=K0, phi_s0=PHI_S0))
	zero_cells = module.Permeability(
		kin, module.Parameters(conductivity_0=K0, phi_s0=PHI_S0, phi_c=dolfin.Constant(0.0), phi_c0=0.0)
	)
	assert mean(two_phase.hydraulic_conductivity, mesh) == pytest.approx(
		mean(zero_cells.hydraulic_conductivity, mesh), rel=1e-12
	)
	assert mean(two_phase.porosity, mesh) == pytest.approx(1.0 - PHI_S0 / 0.8, rel=1e-12)


@pytest.mark.parametrize("module", [kozeny_carman, holmes_mow, compaction_power])
def test_reference_state_without_fluid_is_rejected(module: Any) -> None:
	"""Phi_s0 + Phi_c0 >= 1 (phi_l0 <= 0): refused."""
	with pytest.raises(ValueError):
		module.Parameters(conductivity_0=K0, phi_s0=0.9, phi_c0=0.1)
	with pytest.raises(ValueError):
		module.Parameters(conductivity_0=K0, phi_s0=dolfin.Constant(1.2))


@pytest.mark.parametrize("override", [{"exponent": -1.0}, {"stiffening": -0.5}])
def test_holmes_mow_rejects_negative_constants(override: dict[str, float]) -> None:
	"""kappa_HM < 0 or M < 0: refused."""
	with pytest.raises(ValueError):
		holmes_mow.Parameters(conductivity_0=K0, phi_s0=PHI_S0, **override)
