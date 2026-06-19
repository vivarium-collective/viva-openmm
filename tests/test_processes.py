"""Tests for OpenMMMartiniProcess.

The shape tests run anywhere. The integration test that actually steps the
Martini integrator skips when openmm / martini_openmm aren't installed.
"""

import pytest

from process_bigraph import allocate_core
from pbg_openmm import OpenMMMartiniProcess, bundled_forcefield


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
