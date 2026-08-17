# viva-openmm

Process-bigraph wrapper that runs **real Martini 3 coarse-grained molecular
dynamics** in [OpenMM](https://openmm.org), driven through
[martini_openmm](https://github.com/maccallumlab/martini_openmm) for a correct
Martini setup (reaction-field electrostatics, Martini nonbonded, 310 K, ~20 fs).

It takes a Martini system — a `.gro` + a `.top` with per-molecule `.itp`
includes — and advances it: energy-minimize once, then integrate NVT, emitting
potential/kinetic energy each step and (optionally) a DCD trajectory + final
`.gro`. The headline input is the **parsimony → Martini E. coli slice** produced
by [`pbg-martini`](https://github.com/vivarium-collective/pbg-martini): protein
structures coarse-grained with martinize2 and stamped at positions taken from
the v2ecoli whole-cell model via parsimony packing.

This is a **real bridge** — `update()` drives a genuine `openmm.app.Simulation`
with Martini 3 forces, not a reimplementation. Realism follows Stevens et al.
2023 (*Front. Chem.* 11:1106495), the Martini JCVI-syn3A whole-cell model: same
force field and reaction-field handling; here the molecule positions come from
an *E. coli* WCM rather than Bentopy random packing.

## Install

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
# martini_openmm is GitHub-only (not on PyPI):
uv pip install "git+https://github.com/maccallumlab/martini_openmm.git"
```

OpenMM provides `Reference`, `CPU`, and `OpenCL` platforms; `OpenCL` uses the
GPU (Apple Metal on Apple Silicon) and is much faster for large systems.

Once installed, the process registers automatically via
`bigraph_schema.package.discover` — no manual `register_link()` needed.

## Quick start

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

Running the pbg-martini parsimony slice (≈14.5k CG beads) through this gives,
on OpenCL: potential energy from +1.7×10⁷ kJ/mol (packing clashes) down to a
stable ≈ −1.3×10⁵ kJ/mol after minimize + NVT — a genuine Martini trajectory.

## Topology preparation

A martinize `.top` lists species by *slug* and omits the force field.
`prepare_martini_top` (called automatically when you pass `itp_dir`) fixes both:

- `#include`s the vendored `martini_v3.0.0.itp` first;
- rewrites each `[ molecules ]` entry to the **real `[ moleculetype ]` name**
  inside that species' `.itp` (martinize2 names a multi-chain molecule
  `<slug>_0`, so slug ≠ moleculetype);
- validates the topology particle count against the coordinate file, raising on
  drift (e.g. a multi-chain complex whose stamped bead count differs from its
  itp).

## API

| Port | Dir | Schema | Meaning |
|---|---|---|---|
| `temperature` | in | `float` | thermostat setpoint (K); a sibling may drive it |
| `potential_energy` | out | `overwrite[float]` | current PE (kJ/mol), a sensor reading |
| `kinetic_energy` | out | `overwrite[float]` | current KE (kJ/mol) |
| `n_steps` | out | `integer` | steps integrated this update |

Key config: `gro`, `top`, `itp_dir`, `forcefield_itp`, `temperature`,
`timestep_fs` (20), `nonbonded_cutoff_nm` (1.1), `epsilon_r` (15),
`platform` (`CPU`/`OpenCL`/`Reference`), `minimize`, `min_iterations`,
`steps_per_update`, `dcd_out`, `dcd_interval`.

## Limitations

- **Vacuum slice, not the solvated whole cell.** The paper's realism also
  includes Martini water + ions and the full envelope/chromosome; this runs a
  protein slice in vacuum. Solvation (Martini `W` beads + ions) is the next
  realism step.
- **Multi-chain complexes** (e.g. GroEL) need their stamped bead count to match
  martinize2's moleculetype; `prepare_martini_top` raises on mismatch rather
  than producing a wrong system.
- Coordinates must already be a valid Martini system (this wraps the engine; it
  doesn't coarse-grain — that's pbg-martini's job).

## Tests

```bash
pytest        # 6 tests; the real-MD test skips if openmm/martini_openmm absent
```
