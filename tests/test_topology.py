"""Offline tests for the Martini topology-prep helper (no OpenMM needed)."""

import os

import pytest

from pbg_openmm.topology import (
    prepare_martini_top,
    _moleculetype_name,
    _moleculetype_natoms,
)

# A tiny single-chain itp and a multi-chain one whose moleculetype name differs
# from its slug (the martinize2 `_0` convention that broke the naive .top).
SINGLE_ITP = """\
[ moleculetype ]
prot 1

[ atoms ]
 1 P2 1 ALA BB 1 0.0
 2 SC3 1 ALA SC1 2 0.0
"""

MULTI_ITP = """\
[ moleculetype ]
complex_0 1

[ atoms ]
 1 P2 1 GLY BB 1 0.0
 2 P2 2 GLY BB 2 0.0
 3 P2 3 GLY BB 3 0.0
"""

SRC_TOP = """\
[ system ]
x

[ molecules ]
prot 2
complex 1
"""


def _setup(tmp_path):
    (tmp_path / "prot.itp").write_text(SINGLE_ITP)
    (tmp_path / "complex.itp").write_text(MULTI_ITP)
    (tmp_path / "ff.itp").write_text("; fake ff\n")
    (tmp_path / "system.top").write_text(SRC_TOP)


def test_moleculetype_parsing(tmp_path):
    _setup(tmp_path)
    assert _moleculetype_name(str(tmp_path / "prot.itp")) == "prot"
    assert _moleculetype_name(str(tmp_path / "complex.itp")) == "complex_0"
    assert _moleculetype_natoms(str(tmp_path / "prot.itp")) == 2
    assert _moleculetype_natoms(str(tmp_path / "complex.itp")) == 3


def test_prepare_remaps_names_and_counts(tmp_path):
    _setup(tmp_path)
    out = str(tmp_path / "prepared.top")
    info = prepare_martini_top(
        str(tmp_path / "system.top"), str(tmp_path), out,
        forcefield_itp=str(tmp_path / "ff.itp"),
    )
    # 2 prot (2 atoms each) + 1 complex (3 atoms) = 7 particles
    assert info["n_particles"] == 7
    # the slug 'complex' was remapped to the real moleculetype 'complex_0'
    assert ("complex_0", 1) in info["molecules"]
    assert ("prot", 2) in info["molecules"]
    text = open(out).read()
    assert text.startswith('#include')          # FF included first
    assert "complex_0 1" in text                  # remapped name in [molecules]
    assert "[ molecules ]" in text


def test_expected_natoms_mismatch_raises(tmp_path):
    _setup(tmp_path)
    out = str(tmp_path / "prepared.top")
    # Coordinate file claims 99 particles; topology yields 7 -> must raise.
    with pytest.raises(ValueError, match="inconsistent"):
        prepare_martini_top(
            str(tmp_path / "system.top"), str(tmp_path), out,
            forcefield_itp=str(tmp_path / "ff.itp"),
            expected_natoms=99,
        )
