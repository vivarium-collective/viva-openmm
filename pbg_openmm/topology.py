"""Prepare a runnable Martini system from martinize2 output.

A martinize2 ``.top`` produced by an assembler (e.g. pbg-martini's parsimony
slice) references each species by a *slug* in ``[ molecules ]`` and does not
``#include`` the Martini force-field definitions. martini_openmm needs:

1. the force-field ``.itp`` (``martini_v3.0.0.itp``) included first, so the bead
   types / nonbonded parameters are defined; and
2. each ``[ molecules ]`` entry to match the real ``[ moleculetype ]`` name
   *inside* that species' ``.itp`` -- martinize2 names a multi-chain molecule
   ``<slug>_0`` (etc.), so the slug and the moleculetype name can differ.

:func:`prepare_martini_top` rewrites the topology to satisfy both, resolving the
real moleculetype names and validating that the resulting particle count matches
the coordinate file before anything reaches OpenMM.
"""

from __future__ import annotations

import os
from importlib import resources


def bundled_forcefield() -> str:
    """Absolute path to the vendored Martini 3 force-field ``.itp``."""
    return str(resources.files("pbg_openmm").joinpath("data", "martini_v3.0.0.itp"))


def _moleculetype_name(itp_path: str) -> str | None:
    lines = open(itp_path).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("[ moleculetype ]"):
            for j in range(i + 1, len(lines)):
                s = lines[j].strip()
                if s and not s.startswith(";"):
                    return s.split()[0]
    return None


def _moleculetype_natoms(itp_path: str) -> int:
    """Count atoms in an itp's first ``[ atoms ]`` block."""
    lines = open(itp_path).read().splitlines()
    n, in_atoms = 0, False
    for ln in lines:
        s = ln.strip()
        if s.startswith("[ atoms ]"):
            in_atoms = True
            continue
        if in_atoms:
            if s.startswith("["):
                break
            if s and not s.startswith(";"):
                n += 1
    return n


def _read_molecules(src_top: str):
    """Return ``[(slug, count), ...]`` from a martinize ``.top`` [molecules]."""
    mols, in_mol = [], False
    for ln in open(src_top):
        s = ln.strip()
        if s.startswith("[ molecules ]"):
            in_mol = True
            continue
        if in_mol and s and not s.startswith(";"):
            if s.startswith("["):
                break
            parts = s.split()
            mols.append((parts[0], int(parts[1])))
    return mols


def prepare_martini_top(
    src_top: str,
    itp_dir: str,
    out_top: str,
    forcefield_itp: str | None = None,
    expected_natoms: int | None = None,
    system_name: str = "parsimony martini system",
) -> dict:
    """Rewrite ``src_top`` into an OpenMM-ready Martini topology at ``out_top``.

    Parameters
    ----------
    src_top : str
        martinize2-style ``.top`` whose ``[ molecules ]`` lists species slugs.
    itp_dir : str
        Directory holding the per-species ``.itp`` files (named ``<slug>.itp``).
    out_top : str
        Where to write the prepared topology.
    forcefield_itp : str, optional
        Path to ``martini_v3.0.0.itp``; defaults to the vendored copy.
    expected_natoms : int, optional
        If given (e.g. the coordinate-file particle count), raise when the
        topology's total particle count disagrees -- catches gro/top drift such
        as a multi-chain complex whose stamped bead count != its itp.
    system_name : str
        Name written into the ``[ system ]`` record.

    Returns
    -------
    dict
        ``{out_top, n_particles, molecules: [(moltype, count), ...]}``.
    """
    ff = forcefield_itp or bundled_forcefield()
    itps = [f for f in os.listdir(itp_dir) if f.endswith(".itp")
            and os.path.realpath(os.path.join(itp_dir, f)) != os.path.realpath(ff)]
    slug2mol = {f[:-4]: _moleculetype_name(os.path.join(itp_dir, f)) for f in itps}
    slug2nat = {f[:-4]: _moleculetype_natoms(os.path.join(itp_dir, f)) for f in itps}

    molecules, n_particles = [], 0
    for slug, count in _read_molecules(src_top):
        if count <= 0:
            continue
        moltype = slug2mol.get(slug, slug)
        molecules.append((moltype, count))
        n_particles += slug2nat.get(slug, 0) * count

    if expected_natoms is not None and n_particles != expected_natoms:
        raise ValueError(
            f"topology particle count {n_particles} != coordinate count "
            f"{expected_natoms}; gro/top are inconsistent (a multi-chain "
            f"species' stamped bead count likely differs from its itp). "
            f"molecules={molecules}")

    with open(out_top, "w") as f:
        f.write(f'#include "{ff}"\n')
        for itp in sorted(itps):
            f.write(f'#include "{os.path.join(itp_dir, itp)}"\n')
        f.write(f"\n[ system ]\n{system_name}\n\n[ molecules ]\n")
        for moltype, count in molecules:
            f.write(f"{moltype} {count}\n")

    return {"out_top": out_top, "n_particles": n_particles, "molecules": molecules}
