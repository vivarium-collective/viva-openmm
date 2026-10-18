"""Build a coarse-grained lipid **vesicle** (spherical bilayer shell) as a
Martini ``.gro`` + ``.top`` pair, ready for :class:`OpenMMMartiniProcess`.

The geometry here is force-field-agnostic: a lipid is described only by its
ordered list of bead names and which bead is the head. Lipids are distributed
evenly over two concentric spheres (a Fibonacci lattice, so points don't clump
at the poles) and laid out radially -- outer-leaflet heads point outward, inner-
leaflet heads point inward, tails meet at the bilayer midplane. That makes the
builder reusable for any single- or multi-bead Martini lipid (fatty acids,
phospholipids), with the actual bonded/nonbonded parameters supplied separately
by each lipid's ``.itp``.

A freshly built vesicle is only approximately relaxed; run it under the
``membrane`` barostat (semi-isotropic NPT) so the leaflet area equilibrates
without the periodic box crushing the shell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class LipidSpec:
    """Geometry of one lipid molecule (not its force-field parameters).

    beads: ordered bead names, head first by convention.
    head_index: which bead sits at the water-facing leaflet surface.
    bond_nm: bead-to-bead spacing along the chain (Martini default ~0.47 nm).
    resname: residue/molecule name used in the ``.gro``/``.top``.
    """

    resname: str
    beads: list[str]
    head_index: int = 0
    bond_nm: float = 0.47

    @property
    def length_nm(self) -> float:
        return (len(self.beads) - 1) * self.bond_nm


# Tail C1-bead count per saturated chain (COO head + N C1 beads), matching the
# bundled martini_v3.0.0_fatty_acids_v1.itp (standard Martini ~4:1 mapping).
_FA_TAIL_BEADS = {10: 2, 12: 3, 14: 3, 16: 4}


def fatty_acid_spec(chain: int, *, protonated: bool = True) -> LipidSpec:
    """:class:`LipidSpec` for a bundled Martini 3 saturated fatty acid.

    ``chain`` in {10, 12, 14, 16}. ``protonated=True`` is the neutral COOH
    species (resname ``FAnn``, SP2 head); ``protonated=False`` the deprotonated
    carboxylate (resname ``FAnnM``, Q5 head). Bead layout matches
    ``martini_v3.0.0_fatty_acids_v1.itp``.
    """
    if chain not in _FA_TAIL_BEADS:
        raise ValueError(
            f"no bundled fatty acid for C{chain}; have {sorted(_FA_TAIL_BEADS)}")
    resname = f"FA{chain}" + ("" if protonated else "M")
    beads = ["COO"] + [f"C{i}A" for i in range(1, _FA_TAIL_BEADS[chain] + 1)]
    return LipidSpec(resname=resname, beads=beads, head_index=0, bond_nm=0.47)


def _fibonacci_sphere(n: int) -> list[tuple[float, float, float]]:
    """Return ``n`` unit vectors spread evenly over the sphere."""
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 1.0)]
    pts = []
    ga = math.pi * (3.0 - math.sqrt(5.0))      # golden angle
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1)            # z in [1, -1]
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = ga * i
        pts.append((r * math.cos(theta), r * math.sin(theta), z))
    return pts


def _lipids_for_area(radius_nm: float, area_per_lipid_nm2: float) -> int:
    """How many lipids tile a sphere of this radius at the given APL."""
    return max(1, round(4.0 * math.pi * radius_nm * radius_nm / area_per_lipid_nm2))


@dataclass
class VesicleResult:
    n_outer: int
    n_inner: int
    n_lipid_beads: int
    n_water: int
    radius_nm: float
    box_nm: float
    gro_path: str
    top_path: str
    counts: dict = field(default_factory=dict)

    @property
    def n_lipids(self) -> int:
        return self.n_outer + self.n_inner


def build_vesicle(
    lipid: LipidSpec,
    *,
    radius_nm: float = 8.0,
    leaflet_sep_nm: float = 2.0,
    area_per_lipid_nm2: float = 0.60,
    n_outer: int | None = None,
    n_inner: int | None = None,
    water_resname: str = "W",
    n_water: int = 0,
    water_shell_nm: float = 1.2,
    box_nm: float | None = None,
    out_gro: str = "vesicle.gro",
    out_top: str = "vesicle.top",
    forcefield_itp: str | None = None,
    lipid_itp: str | None = None,
    water_itp: str | None = None,
    title: str = "viva-openmm martini vesicle",
) -> VesicleResult:
    """Build a bilayer vesicle and write ``out_gro`` + ``out_top``.

    ``radius_nm`` is the bilayer midplane radius; the two leaflets sit at
    ``radius ± leaflet_sep/2``. Lipid counts come from ``area_per_lipid_nm2``
    (per-leaflet, scaled by each leaflet's own radius) unless ``n_outer`` /
    ``n_inner`` are given explicitly. Optionally fills ``n_water`` solvent beads
    into the interior lumen and an exterior shell.
    """
    if radius_nm <= leaflet_sep_nm:
        raise ValueError("radius_nm must exceed leaflet_sep_nm")
    r_out = radius_nm + leaflet_sep_nm / 2.0
    r_in = radius_nm - leaflet_sep_nm / 2.0
    if n_outer is None:
        n_outer = _lipids_for_area(r_out, area_per_lipid_nm2)
    if n_inner is None:
        n_inner = _lipids_for_area(r_in, area_per_lipid_nm2)

    chain = lipid.length_nm
    atoms: list[tuple[str, str, tuple[float, float, float]]] = []  # (res, bead, xyz)

    def _place_leaflet(n: int, head_r: float, outward: bool):
        # head at head_r; chain runs toward the midplane. outward=True -> head is
        # the outer surface (chain points inward); False -> head is inner surface.
        sign = -1.0 if outward else +1.0     # direction from head toward tail
        for direction in _fibonacci_sphere(n):
            dx, dy, dz = direction
            for j, bead in enumerate(lipid.beads):
                # distance of bead j from the head along the chain
                along = (j - lipid.head_index) * lipid.bond_nm
                r = head_r + sign * along
                atoms.append((lipid.resname, bead, (r * dx, r * dy, r * dz)))

    _place_leaflet(n_outer, r_out, outward=True)
    _place_leaflet(n_inner, r_in, outward=False)
    n_lipid_beads = len(atoms)

    # Solvent: split between the interior lumen and an exterior shell.
    placed_water = 0
    if n_water > 0:
        r_lumen = r_in - chain - 0.3                      # inside the inner tails
        r_ext = r_out + chain + water_shell_nm            # outside the outer tails
        n_lumen = n_water // 2
        n_ext = n_water - n_lumen
        if r_lumen > 0.5:
            for k, d in enumerate(_fibonacci_sphere(n_lumen)):
                rr = r_lumen * (0.4 + 0.6 * ((k % 5) / 4.0))   # a few nested shells
                atoms.append((water_resname, water_resname,
                              (rr * d[0], rr * d[1], rr * d[2])))
                placed_water += 1
        for d in _fibonacci_sphere(n_ext):
            atoms.append((water_resname, water_resname,
                          (r_ext * d[0], r_ext * d[1], r_ext * d[2])))
            placed_water += 1

    if box_nm is None:
        box_nm = 2.0 * (r_out + chain + water_shell_nm) + 2.0   # padding
    half = box_nm / 2.0

    # --- write .gro (coords in nm, GROMACS fixed-width; shift to box centre) ---
    with open(out_gro, "w") as fh:
        fh.write(title + "\n")
        fh.write(f"{len(atoms)}\n")
        for i, (res, bead, (x, y, z)) in enumerate(atoms):
            fh.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (
                (i + 1) % 100000, res[:5], bead[:5], (i + 1) % 100000,
                x + half, y + half, z + half))
        fh.write("%10.5f%10.5f%10.5f\n" % (box_nm, box_nm, box_nm))

    # --- write .top (contiguous [molecules] blocks matching .gro order) --------
    counts = {lipid.resname: n_outer + n_inner}
    if placed_water:
        counts[water_resname] = placed_water
    with open(out_top, "w") as fh:
        if forcefield_itp:
            fh.write(f'#include "{forcefield_itp}"\n')
        if lipid_itp:
            fh.write(f'#include "{lipid_itp}"\n')
        if water_itp:
            fh.write(f'#include "{water_itp}"\n')
        fh.write("\n[ system ]\nviva-openmm vesicle\n\n[ molecules ]\n")
        fh.write(f"{lipid.resname} {n_outer + n_inner}\n")
        if placed_water:
            fh.write(f"{water_resname} {placed_water}\n")

    return VesicleResult(
        n_outer=n_outer, n_inner=n_inner, n_lipid_beads=n_lipid_beads,
        n_water=placed_water, radius_nm=radius_nm, box_nm=box_nm,
        gro_path=out_gro, top_path=out_top, counts=counts)
