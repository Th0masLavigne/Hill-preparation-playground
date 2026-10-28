"""Unit tests the ``DecimatedTimeIntegrator`` selected fields and rank-0 logs."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from integrator_case import run  # noqa: E402

from hill_playground.io.xdmf_reading import read_time_stamps  # noqa: E402

CASE = Path(__file__).resolve().parent / "integrator_case.py"


@pytest.mark.parametrize("functions", ["foi", "default"])
def test_written_attributes(tmp_path: Path, functions: str) -> None:
	"""``functions_to_write``: only ``double_c`` at every frame; ``None``: subsol, its old copy, FOI."""
	stem = str(tmp_path / functions)
	assert run(stem, functions=functions, n_steps=4)[0]
	text = Path(stem + ".xdmf").read_text()
	names = set(re.findall(r'Attribute Name="([^"]+)"', text))
	assert names == ({"double_c"} if functions == "foi" else {"c", "c_old", "double_c"})
	times = read_time_stamps(Path(stem + ".xdmf"))
	assert times == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0], rel=1e-12, abs=1e-12)


def _numbers_masked(path: Path) -> list[str]:
	"""Lines with every number replaced (timings differ between runs)."""
	return [re.sub(r"[-+]?\d[\d.e+-]*", "N", line) for line in path.read_text().splitlines()]


def _open_mpi_flags() -> list[str]:
	"""Open MPI only (MPICH's hydra rejects them): allow 2 ranks on 1 core, and root in a sandbox."""
	version = subprocess.run(["mpirun", "--version"], capture_output=True, text=True).stdout
	if "Open MPI" not in version:
		return []
	return ["--oversubscribe"] + (["--allow-run-as-root"] if os.geteuid() == 0 else [])


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
def test_mpi_logs_match_serial(tmp_path: Path) -> None:
	"""Under ``mpirun -n 2`` the ``.out``, ``.sta`` and ``.dat`` are those of the serial run."""
	assert run(str(tmp_path / "serial"))[0]
	cmd = ["mpirun", *_open_mpi_flags(), "-n", "2", sys.executable, str(CASE), str(tmp_path / "mpi"), "1"]
	# The parent already initialised MPI (dolfin): drop its singleton variables for the child job.
	env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_"))}
	result = subprocess.run(cmd, capture_output=True, text=True, env=env)
	assert result.returncode == 0, result.stderr[-2000:]
	assert _numbers_masked(tmp_path / "mpi.out") == _numbers_masked(tmp_path / "serial.out")
	assert (tmp_path / "mpi.sta").read_text() == (tmp_path / "serial.sta").read_text()
	np.testing.assert_allclose(
		np.loadtxt(tmp_path / "mpi-qois.dat"), np.loadtxt(tmp_path / "serial-qois.dat"), rtol=1e-10, atol=0.0
	)


def test_silenced_rank_writes_nothing(tmp_path: Path) -> None:
	"""``_silence_logs`` (what non-root ranks do): later log lines no longer reach the files."""
	stem = tmp_path / "silenced"
	_, integrator = run(str(stem), n_steps=2)
	sizes = [Path(f"{stem}{ext}").stat().st_size for ext in (".out", ".sta", "-qois.dat")]
	integrator._silence_logs()
	integrator.printer.print_str("not logged")
	integrator.table_printer.write_line([0] * 7)
	integrator.qoi_printer.write_line([0.0, 0.0])
	integrator.close()
	assert [Path(f"{stem}{ext}").stat().st_size for ext in (".out", ".sta", "-qois.dat")] == sizes


def test_plateau_exit_on_steady_diffusion(tmp_path: Path) -> None:
	"""Diffusion reaches steady state: the step exits early and the last frame is written."""
	stem = tmp_path / "plateau"
	success, _ = run(str(stem), n_steps=40, min_simu_seconds=0.05)  # step 1: before the floor; then not yet stagnant
	assert success
	times = read_time_stamps(Path(f"{stem}.xdmf"))
	assert times[-1] < 1.0
	assert "plateaued" in Path(f"{stem}.out").read_text()


def test_non_root_rank_silences_at_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	"""Rank != 0 at construction: ``_silence_logs`` is called (stdout printers and no QOI file included)."""
	import types

	import hill_playground.integration.decimated_time_integrator as dti

	calls: list[bool] = []
	real = dti.DecimatedTimeIntegrator._silence_logs  # pyright: ignore[reportPrivateUsage]

	def spy(self: object) -> None:
		calls.append(True)
		self.printer.must_close = False  # type: ignore[attr-defined]  # stdout-like printer branch
		self.write_qois = False  # type: ignore[attr-defined]  # no QOI file branch
		real(self)  # type: ignore[arg-type]

	monkeypatch.setattr(dti.DecimatedTimeIntegrator, "_silence_logs", spy)
	monkeypatch.setattr(dti, "MPI", types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(rank=1)))
	with pytest.raises(AttributeError):  # the fake MPI has no allreduce: stop at the first collective
		run(str(tmp_path / "rank1"), n_steps=2)
	assert calls == [True]
