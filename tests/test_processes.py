"""Tests for OpenMMMartiniProcess.

The shape tests run anywhere. The integration test that actually steps the
Martini integrator skips when openmm / martini_openmm aren't installed.
"""

import pytest

from process_bigraph import allocate_core
from viva_openmm import OpenMMMartiniProcess, bundled_forcefield


def test_ports_and_instantiation():
    core = allocate_core()
    p = OpenMMMartiniProcess(config={"gro": "x.gro", "top": "x.top"}, core=core)
    assert isinstance(p.inputs(), dict) and "temperature" in p.inputs()
    outs = p.outputs()
    assert outs["potential_energy"] == "overwrite[float]"
    assert outs["n_steps"] == "integer"


def test_bundled_forcefield_present():
    import os
    ff = bundled_forcefield()
    assert os.path.exists(ff)
    head = open(ff).read(400)
    assert "Martini 3" in head and "[ atomtypes ]" in open(ff).read(2000)


# --- real Martini MD on a minimal 2-bead water system -----------------------

_WATER_ITP = """\
[ moleculetype ]
W 1

[ atoms ]
 1 W 1 W W 1 0.0
"""

_WATER_GRO = """\
two martini waters
2
    1W      W     1   1.000   1.000   1.000
    2W      W     2   1.600   1.000   1.000
   4.00000   4.00000   4.00000
"""

_WATER_TOP = """\
[ system ]
w

[ molecules ]
W 2
"""


def test_real_martini_md_runs(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("martini_openmm")

    (tmp_path / "W.itp").write_text(_WATER_ITP)
    (tmp_path / "system.gro").write_text(_WATER_GRO)
    (tmp_path / "system.top").write_text(_WATER_TOP)

    core = allocate_core()
    p = OpenMMMartiniProcess(config={
        "gro": str(tmp_path / "system.gro"),
        "top": str(tmp_path / "system.top"),
        "itp_dir": str(tmp_path),
        "platform": "Reference",          # deterministic, no GPU needed
        "minimize": True, "min_iterations": 50,
        "steps_per_update": 20,
    }, core=core)

    out = p.update(p.initial_state(), interval=1.0)
    assert out["n_steps"] == 20
    assert isinstance(out["potential_energy"], float)
    # finite (not NaN) energy => the Martini force field really integrated
    assert out["potential_energy"] == out["potential_energy"]

    gro = p.write_final_gro(str(tmp_path / "final.gro"))
    text = open(gro).read()
    assert text.splitlines()[1].strip() == "2"   # 2 particles round-tripped


def _npt_process(tmp_path, **extra):
    """Build an OpenMMMartiniProcess on the 2-water fixture with NPT config."""
    (tmp_path / "W.itp").write_text(_WATER_ITP)
    (tmp_path / "system.gro").write_text(_WATER_GRO)
    (tmp_path / "system.top").write_text(_WATER_TOP)
    cfg = {
        "gro": str(tmp_path / "system.gro"),
        "top": str(tmp_path / "system.top"),
        "itp_dir": str(tmp_path),
        "platform": "Reference",
        "minimize": True, "min_iterations": 50,
        "steps_per_update": 20, "barostat_interval": 10,
    }
    cfg.update(extra)
    return OpenMMMartiniProcess(config=cfg, core=allocate_core())


@pytest.mark.slow
def test_npt_isotropic_barostat_runs(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("martini_openmm")
    p = _npt_process(tmp_path, barostat="isotropic", pressure_bar=1.0)
    out = p.update(p.initial_state(), interval=1.0)        # also drives baro temp
    assert out["n_steps"] == 20
    assert out["potential_energy"] == out["potential_energy"]   # finite
    forces = [type(f).__name__ for f in p._sim.system.getForces()]
    assert "MonteCarloBarostat" in forces


@pytest.mark.slow
def test_npt_membrane_barostat_runs(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("martini_openmm")
    # semi-isotropic (XY coupled, Z free) -- the bilayer/vesicle ensemble
    p = _npt_process(tmp_path, barostat="membrane", surface_tension_bar_nm=0.0)
    out = p.update(p.initial_state(), interval=1.0)
    assert out["n_steps"] == 20
    assert out["potential_energy"] == out["potential_energy"]
    forces = [type(f).__name__ for f in p._sim.system.getForces()]
    assert "MonteCarloMembraneBarostat" in forces


@pytest.mark.slow
def test_unknown_barostat_raises(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("martini_openmm")
    p = _npt_process(tmp_path, barostat="bogus")
    with pytest.raises(ValueError, match="unknown barostat"):
        p.update(p.initial_state(), interval=1.0)
