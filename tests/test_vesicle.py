"""Tests for the Martini vesicle builder (geometry + .gro/.top I/O)."""

import math

import pytest

from viva_openmm.vesicle import LipidSpec, build_vesicle, _fibonacci_sphere


def test_fibonacci_sphere_unit_vectors():
    pts = _fibonacci_sphere(200)
    assert len(pts) == 200
    for x, y, z in pts:
        assert abs(math.sqrt(x * x + y * y + z * z) - 1.0) < 1e-6


def test_build_vesicle_geometry_and_files(tmp_path):
    lipid = LipidSpec(resname="FA12", beads=["COO", "C1", "C2"], head_index=0)
    gro = str(tmp_path / "v.gro")
    top = str(tmp_path / "v.top")
    res = build_vesicle(
        lipid, radius_nm=6.0, leaflet_sep_nm=2.0, area_per_lipid_nm2=0.6,
        out_gro=gro, out_top=top,
        forcefield_itp="martini_v3.0.0.itp", lipid_itp="FA12.itp")

    # both leaflets populated; outer has more lipids than inner (larger radius)
    assert res.n_outer > res.n_inner > 0
    assert res.n_lipid_beads == 3 * (res.n_outer + res.n_inner)

    # .gro: header count line matches the number of atoms written
    lines = open(gro).read().splitlines()
    assert int(lines[1].strip()) == res.n_lipid_beads
    assert len(lines) == res.n_lipid_beads + 3          # title + count + atoms + box

    # heads sit on the two leaflet surfaces (r_out, r_in) about the box centre
    half = res.box_nm / 2.0
    r_out = 6.0 + 1.0
    r_in = 6.0 - 1.0
    head_radii = []
    for ln in lines[2:2 + res.n_lipid_beads]:
        bead = ln[10:15].strip()
        if bead == "COO":
            x = float(ln[20:28]) - half
            y = float(ln[28:36]) - half
            z = float(ln[36:44]) - half
            head_radii.append(math.sqrt(x * x + y * y + z * z))
    assert any(abs(r - r_out) < 1e-2 for r in head_radii)
    assert any(abs(r - r_in) < 1e-2 for r in head_radii)

    # .top molecules block matches the .gro lipid count
    toptext = open(top).read()
    assert f"FA12 {res.n_outer + res.n_inner}" in toptext
    assert '#include "FA12.itp"' in toptext


def test_build_vesicle_with_water(tmp_path):
    lipid = LipidSpec(resname="FA16", beads=["COO", "C1", "C2", "C3"])
    res = build_vesicle(
        lipid, radius_nm=7.0, n_water=400,
        out_gro=str(tmp_path / "v.gro"), out_top=str(tmp_path / "v.top"),
        water_itp="W.itp")
    assert res.n_water > 0
    assert res.counts["W"] == res.n_water


def test_vesicle_gro_parses_in_openmm(tmp_path):
    pytest.importorskip("openmm")
    from openmm.app import GromacsGroFile
    lipid = LipidSpec(resname="FA12", beads=["COO", "C1", "C2"])
    res = build_vesicle(lipid, radius_nm=5.0,
                        out_gro=str(tmp_path / "v.gro"),
                        out_top=str(tmp_path / "v.top"))
    conf = GromacsGroFile(str(tmp_path / "v.gro"))
    assert len(conf.getPositions()) == res.n_lipid_beads
    box = conf.getPeriodicBoxVectors()
    assert box is not None


def test_fatty_acid_spec_matches_itp():
    from viva_openmm import fatty_acid_spec
    s16 = fatty_acid_spec(16, protonated=True)
    assert s16.resname == "FA16"
    assert s16.beads == ["COO", "C1A", "C2A", "C3A", "C4A"]
    s12m = fatty_acid_spec(12, protonated=False)
    assert s12m.resname == "FA12M" and len(s12m.beads) == 4
    with pytest.raises(ValueError):
        fatty_acid_spec(18)


@pytest.mark.slow
@pytest.mark.parametrize("protonated", [True, False])
def test_fatty_acid_vesicle_builds_in_martini_openmm(tmp_path, protonated):
    # The real proof the constructed fatty-acid .itp is valid: the vesicle's
    # .gro/.top loads in martini_openmm and the bead types resolve against the
    # Martini 3 force field to build a real OpenMM System. Both protonation
    # states are exercised so the matched small head pair resolves -- protonated
    # SP2 (neutral COOH) and deprotonated SQ5n (carboxylate COO-).
    pytest.importorskip("openmm")
    pytest.importorskip("martini_openmm")
    import martini_openmm as mo
    from openmm import unit
    from openmm.app import GromacsGroFile
    from viva_openmm import (fatty_acid_spec, bundled_forcefield,
                             bundled_fatty_acids)

    lipid = fatty_acid_spec(16, protonated=protonated)   # palmitic acid / -ate
    gro = str(tmp_path / "ves.gro")
    top = str(tmp_path / "ves.top")
    res = build_vesicle(
        lipid, radius_nm=5.0, out_gro=gro, out_top=top,
        forcefield_itp=bundled_forcefield(),
        lipid_itp=bundled_fatty_acids())

    conf = GromacsGroFile(gro)
    topf = mo.MartiniTopFile(
        top, periodicBoxVectors=conf.getPeriodicBoxVectors(), epsilon_r=15.0)
    system = topf.create_system(nonbonded_cutoff=1.1 * unit.nanometer)
    assert system.getNumParticles() == res.n_lipid_beads
    assert system.getNumParticles() == 5 * res.n_lipids    # COO + 4 C1 each
