r"""Characteristic scales :math:`(L_0, \mu_0, t_0, \rho_0)` and the SI <-> dimensionless conversions.

Base scales (equations.tex, Eq. ``eq:base-scales``):

.. math::
	L_0\ [\mathrm{m}], \qquad \mu_0\ [\mathrm{Pa}], \qquad
	t_0 = \frac{L_0^2 \eta}{k_{\text{gel}} \kappa_{\text{gel}}}\ [\mathrm{s}], \qquad
	\rho_0\ [\mathrm{kg\,m^{-3}}],

with :math:`t_0 = L_0^2/c_v` the gel consolidation time (:math:`c_v = (k_{\text{gel}}/\eta)\kappa_{\text{gel}}`),
built by :meth:`CharacteristicScales.from_consolidation_scale`. :math:`\kappa_{\text{gel}}` is the confined
modulus, not :math:`\mu_0` (shear modulus).

``nondimensionalize_X`` divides by the scale of X, ``redimensionalize_X`` multiplies by it: exact inverses, valid
on floats and UFL expressions. With :math:`R_k = \mathcal N_k \tilde R_k`, :math:`\mathcal N_k > 0`
(Eq. ``eq:nd-principle``), the rescaled problem has the SI solution.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacteristicScales:
	r"""Base scales :math:`L_0` (m), :math:`\mu_0` (Pa), :math:`t_0` (s), :math:`\rho_0` (kg/m^3), all > 0.

	Attributes:
		length (float): :math:`L_0`, in m (sample size or drainage length).
		stress (float): :math:`\mu_0`, in Pa (gel shear modulus).
		time (float): :math:`t_0`, in s; see :meth:`from_consolidation_scale`.
		density (float): :math:`\rho_0`, in kg/m^3 (gel mass density).
	"""

	length: float
	stress: float
	time: float
	density: float

	def __post_init__(self) -> None:
		"""Check that every base scale is strictly positive.

		Raises:
			ValueError: If ``length``, ``stress``, ``time`` or ``density`` is not strictly positive.
		"""
		for field_name in ("length", "stress", "time", "density"):
			value = getattr(self, field_name)
			if value <= 0:
				raise ValueError(f"{field_name} must be strictly positive, got {value}.")

	@classmethod
	def from_consolidation_scale(
		cls,
		length: float,
		stress: float,
		density: float,
		permeability: float,
		viscosity: float,
		consolidation_modulus: float,
	) -> "CharacteristicScales":
		r"""Scales with :math:`t_0 = L_0^2 \eta / (k_{\text{gel}} \kappa_{\text{gel}})` (Eq. ``eq:base-scales``).

		``length``, ``stress`` and ``density`` are passed through; only ``time`` is derived. :math:`\kappa_{\text{gel}}`
		and :math:`\mu_0` are separate arguments (see :attr:`permeability_over_viscosity`).

		Args:
			length (float): :math:`L_0`, in m.
			stress (float): :math:`\mu_0`, in Pa (gel shear modulus).
			density (float): :math:`\rho_0`, in kg/m^3.
			permeability (float): :math:`k_{\text{gel}}`, in m^2 (> 0).
			viscosity (float): :math:`\eta`, in Pa.s (> 0).
			consolidation_modulus (float): :math:`\kappa_{\text{gel}}`, in Pa (> 0): confined modulus of
				:math:`c_v = (k_{\text{gel}}/\eta)\,\kappa_{\text{gel}}`, typically ``gel_kappa``.

		Returns:
			CharacteristicScales: The base scales, with the derived ``time``.

		Raises:
			ValueError: If ``permeability``, ``viscosity`` or ``consolidation_modulus`` is not strictly positive, or
				a base scale fails :meth:`__post_init__`.
		"""
		if permeability <= 0:
			raise ValueError(f"permeability must be strictly positive, got {permeability}.")
		if viscosity <= 0:
			raise ValueError(f"viscosity must be strictly positive, got {viscosity}.")
		if consolidation_modulus <= 0:
			raise ValueError(f"consolidation_modulus must be strictly positive, got {consolidation_modulus}.")
		time = length**2 * viscosity / (permeability * consolidation_modulus)
		return cls(length=length, stress=stress, time=time, density=density)

	@classmethod
	def identity(cls) -> "CharacteristicScales":
		r"""Unit scales :math:`L_0 = \mu_0 = t_0 = \rho_0 = 1`: every conversion is the identity (SI solve).

		Returns:
			CharacteristicScales: ``CharacteristicScales(1, 1, 1, 1)``.
		"""
		return cls(length=1.0, stress=1.0, time=1.0, density=1.0)

	@property
	def velocity(self) -> float:
		"""float: Characteristic velocity :math:`L_0 / t_0`, in metres per second."""
		return self.length / self.time

	@property
	def diffusivity(self) -> float:
		"""float: Characteristic diffusivity :math:`L_0^2 / t_0`, in square metres per second."""
		return self.length**2 / self.time

	@property
	def rate(self) -> float:
		r"""float: Characteristic rate :math:`1 / t_0`, in inverse seconds (rate constants, e.g. :math:`1/\tau`)."""
		return 1.0 / self.time

	@property
	def permeability_over_viscosity(self) -> float:
		r"""float: Characteristic hydraulic conductivity :math:`L_0^2 / (\mu_0 t_0)`, in m^2/(Pa.s).

		With :math:`t_0` from :meth:`from_consolidation_scale`:

		.. math::
			\frac{L_0^2}{\mu_0 t_0} = \frac{k_{gel}}{\eta}\,\frac{\kappa_{gel}}{\mu_0},
			\qquad
			\hat K_{gel} = \frac{k_{gel}/\eta}{L_0^2/(\mu_0 t_0)} = \frac{\mu_0}{\kappa_{gel}},

		so :math:`\hat K_{gel} = 1` only if ``stress == consolidation_modulus`` (e.g. :math:`\mu_0 = 1700` Pa,
		:math:`\kappa_{gel} = 1417` Pa: :math:`\hat K_{gel} = 1.20`).
		"""
		return self.length**2 / (self.stress * self.time)

	def nondimensionalize_length(self, value: float) -> float:
		"""Rescale a length (or displacement) from SI metres to dimensionless length units.

		Args:
			value (float): A length or displacement, in metres.

		Returns:
			float: ``value / self.length``.
		"""
		return value / self.length

	def redimensionalize_length(self, value: float) -> float:
		"""Rescale a dimensionless length (or displacement) back to SI metres.

		Args:
			value (float): A dimensionless length or displacement.

		Returns:
			float: ``value * self.length``.
		"""
		return value * self.length

	def nondimensionalize_stress(self, value: float) -> float:
		"""Rescale a stress (or pressure) from SI pascals to dimensionless stress units.

		Args:
			value (float): A stress or pressure, in pascals.

		Returns:
			float: ``value / self.stress``.
		"""
		return value / self.stress

	def redimensionalize_stress(self, value: float) -> float:
		"""Rescale a dimensionless stress (or pressure) back to SI pascals.

		Args:
			value (float): A dimensionless stress or pressure.

		Returns:
			float: ``value * self.stress``.
		"""
		return value * self.stress

	def nondimensionalize_time(self, value: float) -> float:
		"""Rescale a duration from SI seconds to dimensionless time units.

		Args:
			value (float): A duration, in seconds.

		Returns:
			float: ``value / self.time``.
		"""
		return value / self.time

	def redimensionalize_time(self, value: float) -> float:
		"""Rescale a dimensionless duration back to SI seconds.

		Args:
			value (float): A dimensionless duration.

		Returns:
			float: ``value * self.time``.
		"""
		return value * self.time

	def nondimensionalize_rate(self, value: float) -> float:
		"""Rescale a first-order rate constant from SI inverse-seconds to dimensionless units.

		Args:
			value (float): A rate constant, in inverse seconds.

		Returns:
			float: ``value / self.rate``.
		"""
		return value / self.rate

	def redimensionalize_rate(self, value: float) -> float:
		"""Rescale a dimensionless rate constant back to SI inverse-seconds.

		Args:
			value (float): A dimensionless rate constant.

		Returns:
			float: ``value * self.rate``.
		"""
		return value * self.rate

	def nondimensionalize_diffusivity(self, value: float) -> float:
		"""Rescale a diffusivity from SI square-metres-per-second to dimensionless units.

		Args:
			value (float): A diffusion coefficient, in square metres per second.

		Returns:
			float: ``value / self.diffusivity``.
		"""
		return value / self.diffusivity

	def redimensionalize_diffusivity(self, value: float) -> float:
		"""Rescale a dimensionless diffusivity back to SI square-metres-per-second.

		Args:
			value (float): A dimensionless diffusion coefficient.

		Returns:
			float: ``value * self.diffusivity``.
		"""
		return value * self.diffusivity

	def nondimensionalize_permeability_over_viscosity(self, value: float) -> float:
		r"""Rescale a hydraulic conductivity (permeability/viscosity) to dimensionless units.

		Args:
			value (float): Hydraulic conductivity :math:`k/\eta`, in m^2/(Pa.s).

		Returns:
			float: ``value / self.permeability_over_viscosity``.
		"""
		return value / self.permeability_over_viscosity

	def redimensionalize_permeability_over_viscosity(self, value: float) -> float:
		"""Rescale a dimensionless hydraulic conductivity back to SI units.

		Args:
			value (float): A dimensionless hydraulic conductivity.

		Returns:
			float: ``value * self.permeability_over_viscosity``.
		"""
		return value * self.permeability_over_viscosity

	def nondimensionalize_density(self, value: float) -> float:
		"""Rescale a mass density from SI kilograms-per-cubic-metre to dimensionless units.

		Args:
			value (float): A mass density, in kilograms per cubic metre.

		Returns:
			float: ``value / self.density``.
		"""
		return value / self.density

	def redimensionalize_density(self, value: float) -> float:
		"""Rescale a dimensionless mass density back to SI kilograms-per-cubic-metre.

		Args:
			value (float): A dimensionless mass density.

		Returns:
			float: ``value * self.density``.
		"""
		return value * self.density
