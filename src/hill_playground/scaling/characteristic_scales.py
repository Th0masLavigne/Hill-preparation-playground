r"""Characteristic scales for nondimensionalizing the porous microtissue-pillar problem.

Status:
	Promoted copy (plus ``identity()`` and corrected docstrings) of the demo-side ``CharacteristicScales`` of fenicsoft
	(``demos/smooth_muscle_cells/porous_hydrogel/microtissue_pillar_spherical_scale/src``,
	branch ``64-reproduce-experiment-from-mostert-et-al-2``), as that module's own status note
	asks. Target: ``fenicsoft.scaling``.

This module provides the :class:`CharacteristicScales` dataclass: four base characteristic
scales (length, stress, time, density) and the paired ``nondimensionalize_*``/
``redimensionalize_*`` conversions derived from them, used to rescale every residual block of
the monolithic triphasic poromechanics problem so its Jacobian's diagonal spans a much smaller
range of orders of magnitude than the raw-SI system does.

**Why four base scales, and why this particular time scale.** Length (:math:`L_0`), stress
(:math:`\mu_0`) and density (:math:`\rho_0`) are free choices (typically the sample's own
radius, the gel's shear modulus, and the gel's mass density). The time scale :math:`t_0` is
*not* free: it is fixed by requiring the gel's own Darcy (pressure-diffusion) block to become
:math:`O(1)` after rescaling, i.e. the gel's own consolidation time,

.. math::
	t_0 = \frac{L_0^2 \eta}{k_{\text{gel}} \kappa_{\text{gel}}}

where :math:`\eta` is the fluid viscosity, :math:`k_{\text{gel}}` the gel's permeability, and
:math:`\kappa_{\text{gel}}` the gel's own *confined* (bulk/oedometric) modulus -- the
physically-correct consolidation-coefficient modulus (:math:`c_v\sim(k/\eta)\kappa`), which is
**not** the same as :math:`\mu_0` (typically the gel's own *shear* modulus, used as the general
stress scale everywhere else). See :meth:`CharacteristicScales.from_consolidation_scale`, whose
own ``consolidation_modulus`` argument is this :math:`\kappa_{\text{gel}}`.

**Values.** The conversions only multiply or divide, so ``value`` may be a float or a UFL
expression (e.g. ``redimensionalize_stress(p_hat)`` for an exported field).

**Round-trip equivalence.** Every ``nondimensionalize_*``/``redimensionalize_*`` pair is an
exact inverse of the other (each characteristic scale is a strictly positive float, so division
then multiplication -- or the reverse -- recovers the original value up to floating-point
round-off only). This is what makes solving the rescaled problem and then redimensionalizing
its solution for export provably equivalent to solving the original SI problem directly: since
every residual block satisfies :math:`R = \mathcal{N} \hat{R}` for some block-specific,
strictly positive normalization constant :math:`\mathcal{N} \neq 0`, :math:`\hat{R} = 0
\Leftrightarrow R = 0` termwise, and the primal unknown recovered by redimensionalizing the
rescaled solution is exactly the SI solution.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacteristicScales:
	r"""Four base characteristic scales used to nondimensionalize the monolithic residual.

	Attributes:
		length (float): Characteristic length :math:`L_0`, in metres. Typically the sample's own
			overall radius or a representative pillar spacing.
		stress (float): Characteristic stress :math:`\mu_0`, in pascals. Typically the gel's own
			shear modulus.
		time (float): Characteristic time :math:`t_0`, in seconds. Not a free choice in practice
			-- see :meth:`from_consolidation_scale`, which derives it from the gel's own
			consolidation time so the Darcy block becomes :math:`O(1)`.
		density (float): Characteristic density :math:`\rho_0`, in kilograms per cubic metre.
			Typically the gel's own mass density.
	"""

	length: float
	stress: float
	time: float
	density: float

	def __post_init__(self) -> None:
		"""Validate that every characteristic scale is strictly positive.

		Raises:
			ValueError: If ``length``, ``stress``, ``time``, or ``density`` is not strictly
				positive.
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
		r"""Build a :class:`CharacteristicScales` whose time scale is the gel's own consolidation time.

		Derives :math:`t_0 = L_0^2 \eta / (k_{\text{gel}} \kappa_{\text{gel}})` -- the unique
		time scale for which the gel's Darcy (pressure-diffusion) residual block becomes
		:math:`O(1)` after rescaling (see the module docstring). Note that the modulus entering
		this formula is ``consolidation_modulus`` (:math:`\kappa_{\text{gel}}`, the gel's own
		*confined* modulus), which is deliberately a separate argument from ``stress``
		(:math:`\mu_0`, typically the gel's *shear* modulus): the two coincide only if the
		caller passes the same value for both, which is not assumed here. ``length``, ``stress``
		and ``density`` are passed through unchanged; only ``time`` is derived.

		Args:
			length (float): Characteristic length :math:`L_0`, in metres.
			stress (float): Characteristic stress :math:`\mu_0`, in pascals -- the scale used to
				nondimensionalize every stress/pressure field. Typically the gel's shear modulus.
			density (float): Characteristic density :math:`\rho_0`, in kilograms per cubic metre.
			permeability (float): The gel's intrinsic permeability :math:`k_{\text{gel}}`, in
				square metres. Must be strictly positive.
			viscosity (float): The interstitial fluid's dynamic viscosity :math:`\eta`, in
				pascal-seconds. Must be strictly positive.
			consolidation_modulus (float): The gel's own confined/oedometric modulus
				:math:`\kappa_{\text{gel}}`, in pascals -- the modulus entering the
				consolidation-coefficient estimate :math:`c_v\sim(k_{\text{gel}}/\eta)\,
				\kappa_{\text{gel}}`. Typically the gel's own volumetric-penalty bulk modulus
				(``gel_kappa``), *not* necessarily the same value as ``stress``. Must be strictly
				positive.

		Returns:
			CharacteristicScales: The four base scales, with ``time`` set to the derived
				consolidation time.

		Raises:
			ValueError: If ``permeability``, ``viscosity``, or ``consolidation_modulus`` is not
				strictly positive (or if any base scale fails :meth:`__post_init__`'s own
				validation).
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
		r"""float: Characteristic rate :math:`1 / t_0`, in inverse seconds.

		Use this to nondimensionalize any first-order rate constant (a proliferation rate, a
		remodeling relaxation rate :math:`1/\tau`, a haptotaxis sensitivity's own time part).
		"""
		return 1.0 / self.time

	@property
	def permeability_over_viscosity(self) -> float:
		r"""float: Characteristic hydraulic conductivity :math:`L_0^2 / (\mu_0 t_0)`.

		With ``time`` from :meth:`from_consolidation_scale`
		(:math:`t_0 = L_0^2\eta/(k_{gel}\kappa_{gel})`):

		.. math::
			\frac{L_0^2}{\mu_0 t_0} = \frac{k_{gel}}{\eta}\,\frac{\kappa_{gel}}{\mu_0},
			\qquad
			\hat K_{gel} = \frac{k_{gel}/\eta}{L_0^2/(\mu_0 t_0)} = \frac{\mu_0}{\kappa_{gel}}

		(equations.tex, Table tab:groups: :math:`\hat K_{gel} = 1.20`). Both equal
		:math:`k_{gel}/\eta` and 1 only when ``stress == consolidation_modulus``.

		Note:
		    Docstring corrected from the fenicsoft demo-side original, which stated
		    :math:`(k/\eta)\,\mu_0/\kappa` (ratio inverted); the code is unchanged.
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
			value (float): A hydraulic conductivity :math:`k/\eta`, in square metres per
				pascal-second.

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
