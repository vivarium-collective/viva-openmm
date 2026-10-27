# viva-openmm

Process-bigraph molecular-dynamics tooling built on [OpenMM](https://openmm.org).
It provides three things for the [vivarium / process-bigraph](https://github.com/vivarium-collective/process-bigraph)
ecosystem:

1. **`OpenMMMartiniProcess`** — a Process that runs **real Martini 3
   coarse-grained MD**, driven through
   [martini_openmm](https://github.com/maccallumlab/martini_openmm) for a correct
   Martini setup (reaction-field electrostatics, Martini nonbonded, 310 K,
   ~20 fs).
2. **Solvated all-atom relax** — `relax_in_water` / `SolvatedRelaxStep`: minimize
   + briefly equilibrate a protein structure in explicit AMBER14/TIP3P water,
   then strip solvent and write the relaxed protein.
3. **Martini vesicle builder** — `build_vesicle` / `LipidSpec` / `fatty_acid_spec`:
   assemble a coarse-grained bilayer vesicle (`.gro` + `.top`) ready to run
   through `OpenMMMartiniProcess`.

Everything here is a **real bridge** — the Martini process drives a genuine
`openmm.app.Simulation` with Martini 3 forces, and the relax path runs a real
OpenMM solvated simulation. Neither reimplements the physics.

## Install

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
# martini_openmm is GitHub-only (not on PyPI); needed for the Martini process:
uv pip install "git+https://github.com/maccallumlab/martini_openmm.git"
```

Requires Python >= 3.10. OpenMM provides `Reference`, `CPU`, and `OpenCL`
platforms; `OpenCL` uses the GPU (Apple Metal on Apple Silicon) and is much
faster for large systems.

Once installed, `OpenMMMartiniProcess` registers automatically via
`bigraph_schema.package.discover` — no manual `register_link()` needed.

## Martini coarse-grained MD

`OpenMMMartiniProcess` takes a Martini system — a `.gro` + a `.top` with
per-molecule `.itp` includes — and advances it: energy-minimize once, then
integrate (NVT by default), emitting potential/kinetic energy each step and
(optionally) a DCD trajectory + final `.gro`. A typical input is the
parsimony → Martini *E. coli* slice produced by
[`pbg-martini`](https://github.com/vivarium-collective/pbg-martini): protein
structures coarse-grained with martinize2 and stamped at positions taken from the
v2ecoli whole-cell model via parsimony packing. Realism follows Stevens et al.
2023 (*Front. Chem.* 11:1106495), the Martini JCVI-syn3A whole-cell model (same
force field and reaction-field handling).

```python
from process_bigraph import allocate_core
from viva_openmm import OpenMMMartiniProcess

core = allocate_core()
proc = OpenMMMartiniProcess(config={
    "gro": "system.gro",
    "top": "system.top",          # martinize .top; itp_dir resolves the rest
    "itp_dir": "templates",       # per-molecule .itp files
    "platform": "OpenCL",
    "temperature": 310.0,
    "steps_per_update": 500,
    "dcd_out": "traj.dcd",
}, core=core)

out = proc.update(proc.initial_state(), interval=1.0)   # builds, minimizes, 500 steps
print(out["potential_energy"], "kJ/mol")
proc.write_final_gro("final.gro")
```

### Ports

| Port | Dir | Schema | Meaning |
|---|---|---|---|
| `temperature` | in | `float` | thermostat setpoint (K); a sibling may drive it |
| `potential_energy` | out | `overwrite[float]` | current PE (kJ/mol), a sensor reading |
| `kinetic_energy` | out | `overwrite[float]` | current KE (kJ/mol) |
| `n_steps` | out | `integer` | steps integrated this update (additive) |

### Key config

`gro`, `top`, `itp_dir`, `forcefield_itp`, `temperature`, `timestep_fs` (20),
`friction_per_ps`, `nonbonded_cutoff_nm` (1.1), `epsilon_r` (15),
`platform` (`CPU`/`OpenCL`/`Reference`), `minimize`, `min_iterations`,
`steps_per_update`, `dcd_out`, `dcd_interval`, and a gentle post-minimize ramp
(`equil_steps`, `equil_timestep_fs`). The ensemble is set by `barostat`:
`"none"` → NVT (default), `"isotropic"` → NPT (MonteCarloBarostat),
`"membrane"` → semi-isotropic NPT (MonteCarloMembraneBarostat at
`surface_tension_bar_nm`) for a bilayer/vesicle whose area must relax without the
box crushing the curved shell.

### Topology preparation

A martinize `.top` lists species by *slug* and omits the force field.
`prepare_martini_top` (called automatically when you pass `itp_dir`) fixes both:

- `#include`s the vendored `martini_v3.0.0.itp` first;
- rewrites each `[ molecules ]` entry to the **real `[ moleculetype ]` name**
  inside that species' `.itp` (martinize2 names a multi-chain molecule
  `<slug>_0`, so slug ≠ moleculetype);
- validates the topology particle count against the coordinate file, raising on
  drift (e.g. a multi-chain complex whose stamped bead count differs from its
  itp).

## Solvated all-atom relax

Relax an all-atom protein structure in explicit water before coarse-graining or
analysis. `relax_in_water` solvates the structure (AMBER14 + a TIP3P-family water
by default), minimizes and briefly equilibrates under Langevin dynamics, then
strips water/ions and writes the relaxed **protein-only** coordinates. It returns
a provenance dict (every parameter + the OpenMM version) so callers can cite
exactly how the structure was produced.

```python
from viva_openmm import relax_in_water

provenance = relax_in_water("protein.pdb", "protein.relaxed.pdb",
                            padding_nm=1.0, ionic_strength_M=0.15,
                            temperature_K=300.0, equil_ps=200.0)
```

`SolvatedRelaxStep` wraps the same routine as a process-bigraph `Step`
(`structure_path` in → `relaxed_path` out) for use inside a composite.

## Martini vesicle builder

`build_vesicle` assembles a coarse-grained bilayer vesicle and writes a `.gro` +
`.top` you can run directly through `OpenMMMartiniProcess` (use `barostat="membrane"`).
Lipid counts default from an area-per-lipid model scaled to each leaflet's radius,
and the interior lumen / exterior shell can optionally be filled with Martini
water beads.

```python
from viva_openmm import build_vesicle, fatty_acid_spec

lipid = fatty_acid_spec(16)          # bundled Martini 3 C16 saturated fatty acid
result = build_vesicle(lipid, radius_nm=8.0, n_water=0,
                       out_gro="vesicle.gro", out_top="vesicle.top")
print(result.n_lipids, result.n_lipid_beads)
```

`fatty_acid_spec(chain)` supports bundled saturated fatty acids C10/C12/C14/C16
(protonated COOH or deprotonated carboxylate), matching the vendored
`martini_v3.0.0_fatty_acids_v1.itp`. For other lipids, construct a `LipidSpec`
directly.

## Trajectory / crowded-cell viewers

`scripts/make_traj_viewer.py` and `scripts/make_crowded_viewer.py` build
self-contained HTML viewers that animate a Martini DCD trajectory (three.js
instanced spheres, colored by species, with play/pause + frame slider); the
crowded-cell variant hides bulk water by default and renders the solute.

## API overview

Exported from the top-level `viva_openmm` package:

- **`OpenMMMartiniProcess`** — the Martini coarse-grained MD Process.
- **`prepare_martini_top`** — prepare a martinize `.top` (force field include,
  moleculetype rewrite, count validation).
- **`bundled_forcefield`** / **`bundled_fatty_acids`** — paths to the vendored
  Martini 3 `.itp` files.
- **`relax_in_water`** / **`SolvatedRelaxStep`** / **`RelaxError`** — explicit-solvent
  relax (function, process-bigraph Step, and error type).
- **`build_vesicle`** / **`VesicleResult`** / **`LipidSpec`** / **`fatty_acid_spec`** —
  the Martini vesicle builder and its geometry types.

A back-compatibility shim distribution (`pbg-openmm`, under `shim/`) re-exports
`viva_openmm` for the package's former name.

## Scope

- `OpenMMMartiniProcess` **wraps** the Martini engine; it does not coarse-grain —
  that is `pbg-martini`'s job. Coordinates must already be a valid Martini system.
- **Multi-chain complexes** (e.g. GroEL) need their stamped bead count to match
  martinize2's moleculetype; `prepare_martini_top` raises on a mismatch rather
  than producing a wrong system.

## Tests

```bash
pytest        # fast tests; real-MD tests are marked `slow` and excluded by default
```

The `slow`-marked tests run actual OpenMM and skip if `openmm` / `martini_openmm`
are absent.

## Links

- Repository: https://github.com/vivarium-collective/viva-openmm
- OpenMM: https://openmm.org
- martini_openmm: https://github.com/maccallumlab/martini_openmm
- process-bigraph: https://github.com/vivarium-collective/process-bigraph
