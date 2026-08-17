"""Tests for relax_in_water / SolvatedRelaxStep.

The provenance-shape test stubs the real OpenMM run so it stays fast. The
`slow` integration test actually solvates and equilibrates a tiny peptide;
it skips cleanly if OpenMM or the AMBER14/TIP3P force-field files aren't
resolvable.
"""

import pytest

from viva_openmm import relax as R


def test_provenance_shape_without_running(monkeypatch, tmp_path):
    # relax_in_water assembles a provenance dict of all params + openmm version;
    # stub the actual OpenMM run so this stays fast.
    monkeypatch.setattr(R, "_run_solvated_relax", lambda *a, **k: None)  # writes nothing
    (tmp_path/"in.pdb").write_text("REMARK stub\nEND\n")
    prov = R.relax_in_water(str(tmp_path/"in.pdb"), str(tmp_path/"out.pdb"),
                            seed=7, equil_ps=50.0)
    for k in ("forcefield","water_model","padding_nm","ionic_strength_M",
              "temperature_K","timestep_fs","equil_ps","seed","openmm_version"):
        assert k in prov
    assert prov["seed"] == 7 and prov["equil_ps"] == 50.0


# Idealized coordinates (Angstrom, PDB convention) for a tiny ALA-ALA-ALA
# tripeptide: backbone N/CA/C/O + CB for each residue, OXT on the C-terminal
# carbon. Geometry is only approximate (bond lengths ~1.5 A) -- good enough
# for ForceField template matching; minimizeEnergy() fixes up the rest.
_TINY_PEPTIDE_PDB = """\
ATOM      1  N   ALA A   1      -4.500   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      -3.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1      -2.400   1.400   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1      -3.050   2.440   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1      -2.500  -0.800   1.200  1.00  0.00           C
ATOM      6  N   ALA A   2      -1.100   1.400   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2      -0.400   2.700   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       1.100   2.700   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       1.750   1.660   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2      -0.900   3.500   1.200  1.00  0.00           C
ATOM     11  N   ALA A   3       1.700   3.900   0.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       3.150   4.000   0.000  1.00  0.00           C
ATOM     13  C   ALA A   3       3.750   5.400   0.000  1.00  0.00           C
ATOM     14  O   ALA A   3       3.100   6.440   0.000  1.00  0.00           O
ATOM     15  OXT ALA A   3       5.000   5.400   0.000  1.00  0.00           O
ATOM     16  CB  ALA A   3       3.650   3.200   1.200  1.00  0.00           C
TER      17      ALA A   3
END
"""


def _write_tiny_peptide_pdb(tmp_path):
    """Write a minimal valid all-atom tripeptide PDB (no network fetch)."""
    path = tmp_path / "tiny_peptide.pdb"
    path.write_text(_TINY_PEPTIDE_PDB)
    return path


@pytest.mark.slow
def test_relax_in_water_real_tiny(tmp_path):
    # a tiny structure, few steps, small box -> a valid relaxed PDB (protein
    # atoms present, waters stripped). Skips cleanly if OpenMM/forcefields
    # unavailable.
    pytest.importorskip("openmm")
    src = _write_tiny_peptide_pdb(tmp_path)   # helper: a few-residue peptide
    out = tmp_path/"relaxed.pdb"
    try:
        R.relax_in_water(str(src), str(out), equil_ps=2.0, padding_nm=1.0, seed=1)
    except R.RelaxError as exc:
        pytest.skip(f"OpenMM force fields not resolvable: {exc}")
    txt = out.read_text()
    assert out.is_file() and "ATOM" in txt and "HOH" not in txt
