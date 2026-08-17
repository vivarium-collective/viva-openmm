"""Process-bigraph wrapper for OpenMM running Martini coarse-grained MD.

``OpenMMMartiniProcess`` is a **real bridge**: its ``update()`` drives a genuine
``openmm.app.Simulation`` whose forces come from the Martini 3 force field, built
via ``martini_openmm`` (reaction-field electrostatics, Martini nonbonded). It
takes a Martini coordinate file (``.gro``) + topology (``.top`` with per-molecule
``.itp`` includes) -- e.g. pbg-martini's parsimony->Martini E. coli slice -- and
advances it: it energy-minimizes once, then integrates NVT, emitting potential
energy and (optionally) a DCD trajectory.

Realism follows Stevens et al. 2023 (Front. Chem. 11:1106495), the Martini
JCVI-syn3A whole-cell model: same force field, reaction-field handling, 310 K,
~20 fs timestep. Here the molecule positions come from the v2ecoli whole-cell
model via parsimony packing rather than Bentopy.

Heavy dependencies (openmm, martini_openmm) are imported lazily inside
``_build`` so the module imports without them; the test that actually steps the
integrator skips when they're absent.
"""

from __future__ import annotations

import os

from process_bigraph import Process

from .topology import prepare_martini_top


class OpenMMMartiniProcess(Process):
    """Run Martini coarse-grained MD in OpenMM, one chunk of steps per update.

    Inputs
    ------
    temperature : float
        Thermostat setpoint (K). A sibling process (e.g. an environment or
        controller) may drive it; defaults to the configured temperature.

    Outputs
    -------
    potential_energy : overwrite[float]
        Current potential energy (kJ/mol) -- an absolute sensor reading.
    kinetic_energy : overwrite[float]
        Current kinetic energy (kJ/mol).
    n_steps : integer
        Steps integrated this update (additive).
    """

    config_schema = {
        "gro": {"_type": "string", "_default": ""},
        # Either a prepared `.top`, or a martinize `.top` + `itp_dir` to prepare.
        "top": {"_type": "string", "_default": ""},
        "itp_dir": {"_type": "string", "_default": ""},
        "forcefield_itp": {"_type": "string", "_default": ""},
        "temperature": {"_type": "float", "_default": 310.0},
        "timestep_fs": {"_type": "float", "_default": 20.0},
        "friction_per_ps": {"_type": "float", "_default": 1.0},
        "nonbonded_cutoff_nm": {"_type": "float", "_default": 1.1},
        "epsilon_r": {"_type": "float", "_default": 15.0},
        "platform": {"_type": "string", "_default": "CPU"},
        "minimize": {"_type": "boolean", "_default": True},
        "min_iterations": {"_type": "integer", "_default": 500},
        # gentle equilibration ramp after minimize (essential for freshly
        # packed/solvated crowded systems, which blow up at full dt otherwise):
        # run `equil_steps` at `equil_timestep_fs` before production.
        "equil_steps": {"_type": "integer", "_default": 0},
        "equil_timestep_fs": {"_type": "float", "_default": 2.0},
        "steps_per_update": {"_type": "integer", "_default": 500},
        "dcd_out": {"_type": "string", "_default": ""},
        "dcd_interval": {"_type": "integer", "_default": 0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._sim = None
        self._unit = None

    def inputs(self):
        return {"temperature": "float"}

    def outputs(self):
        return {
            "potential_energy": "overwrite[float]",
            "kinetic_energy": "overwrite[float]",
            "n_steps": "integer",
        }

    def initial_state(self):
        return {"temperature": float(self.config["temperature"])}

    # -- real bridge ---------------------------------------------------------
    def _build(self):
        import martini_openmm as mo
        from openmm import app, unit, LangevinMiddleIntegrator, Platform
        self._unit = unit

        gro = self.config["gro"]
        if not gro or not os.path.exists(gro):
            raise FileNotFoundError(f"gro not found: {gro!r}")

        conf = app.GromacsGroFile(gro)
        box = conf.getPeriodicBoxVectors()

        top_path = self.config["top"]
        if self.config["itp_dir"]:
            # Prepare a runnable top from a martinize .top + itp dir.
            prepared = os.path.join(
                os.path.dirname(os.path.abspath(gro)) or ".", "_prepared.top")
            prepare_martini_top(
                top_path, self.config["itp_dir"], prepared,
                forcefield_itp=self.config["forcefield_itp"] or None,
                expected_natoms=len(conf.getPositions()),
            )
            top_path = prepared
        if not top_path or not os.path.exists(top_path):
            raise FileNotFoundError(f"top not found: {top_path!r}")

        top = mo.MartiniTopFile(
            top_path, periodicBoxVectors=box,
            epsilon_r=float(self.config["epsilon_r"]))
        system = top.create_system(
            nonbonded_cutoff=float(self.config["nonbonded_cutoff_nm"]) * unit.nanometer)

        integrator = LangevinMiddleIntegrator(
            float(self.config["temperature"]) * unit.kelvin,
            float(self.config["friction_per_ps"]) / unit.picosecond,
            float(self.config["timestep_fs"]) * unit.femtosecond)
        platform = Platform.getPlatformByName(self.config["platform"])
        sim = app.Simulation(top.topology, system, integrator, platform)
        sim.context.setPositions(conf.getPositions())

        if self.config["minimize"]:
            sim.minimizeEnergy(maxIterations=int(self.config["min_iterations"]))

        # Equilibration ramp: warm up at a small timestep so residual packing
        # overlaps relax without integrator blow-up, then restore production dt.
        equil = int(self.config["equil_steps"])
        if equil > 0:
            prod_dt = integrator.getStepSize()
            integrator.setStepSize(
                float(self.config["equil_timestep_fs"]) * unit.femtosecond)
            sim.context.setVelocitiesToTemperature(
                float(self.config["temperature"]) * unit.kelvin)
            sim.step(equil)
            integrator.setStepSize(prod_dt)

        if self.config["dcd_out"]:
            interval = int(self.config["dcd_interval"]) or int(self.config["steps_per_update"])
            sim.reporters.append(app.DCDReporter(self.config["dcd_out"], interval))

        self._sim = sim
        self._integrator = integrator

    def update(self, state, interval):
        if self._sim is None:
            self._build()

        # A sibling may drive the thermostat setpoint.
        temp = state.get("temperature")
        if temp:
            self._integrator.setTemperature(float(temp) * self._unit.kelvin)

        n = int(self.config["steps_per_update"])
        self._sim.step(n)

        st = self._sim.context.getState(getEnergy=True)
        pe = st.getPotentialEnergy().value_in_unit(self._unit.kilojoule_per_mole)
        ke = st.getKineticEnergy().value_in_unit(self._unit.kilojoule_per_mole)
        return {"potential_energy": pe, "kinetic_energy": ke, "n_steps": n}

    def write_final_gro(self, path: str):
        """Write the current positions as a ``.gro`` (post-run snapshot).

        ``openmm.app.GromacsGroFile`` is read-only, so format the record by hand
        (coordinates in nm, GROMACS fixed-width layout).
        """
        st = self._sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        pos = st.getPositions().value_in_unit(self._unit.nanometer)
        box = st.getPeriodicBoxVectors().value_in_unit(self._unit.nanometer)
        atoms = list(self._sim.topology.atoms())
        with open(path, "w") as fh:
            fh.write("viva-openmm martini final frame\n")
            fh.write(f"{len(pos)}\n")
            for i, (a, p) in enumerate(zip(atoms, pos)):
                fh.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (
                    (a.residue.index + 1) % 100000, a.residue.name[:5],
                    a.name[:5], (i + 1) % 100000, p[0], p[1], p[2]))
            fh.write("%10.5f%10.5f%10.5f\n" % (box[0][0], box[1][1], box[2][2]))
        return path
