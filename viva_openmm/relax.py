"""All-atom explicit-water MD relaxation of a protein structure.

``relax_in_water`` solvates a PDB in explicit water/ions, briefly equilibrates
it with OpenMM (minimize + short Langevin dynamics), then writes out the
protein-only relaxed coordinates. Intended for compacting disordered protein
tails before downstream 3D packing (parsimony), which is more forgiving of a
locally sensible conformation than of an extended/unphysical one.

The actual OpenMM work is factored into the module-level
``_run_solvated_relax`` helper so tests can monkeypatch it and stay fast; only
``relax_in_water`` builds the provenance dict.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from process_bigraph import Step

# Standard water/ion residue names to exclude when writing the protein-only
# relaxed structure back out.
_SOLVENT_RESIDUE_NAMES = {
    "HOH", "WAT", "TIP", "TIP3", "TIP4", "SPC",
    "NA", "CL", "K", "MG", "CA", "ZN", "SOD", "CLA", "POT",
}


class RelaxError(Exception):
    """Raised when the OpenMM solvated-relaxation run fails."""


def _run_solvated_relax(
    pdb_in: str,
    pdb_out: str,
    *,
    forcefield: Tuple[str, ...],
    water_model: str,
    padding_nm: float,
    ionic_strength_M: float,
    temperature_K: float,
    timestep_fs: float,
    equil_ps: float,
    minimize: bool,
    seed: int,
) -> None:
    """Run the real OpenMM solvated relaxation and write the protein-only PDB.

    Factored out of ``relax_in_water`` so tests can monkeypatch it (and skip
    the real, slow OpenMM run) while still exercising provenance assembly.
    """
    try:
        from openmm import LangevinMiddleIntegrator, Vec3, unit
        from openmm.app import PME, ForceField, HBonds, Modeller, PDBFile, Simulation, Topology

        pdb = PDBFile(pdb_in)
        modeller = Modeller(pdb.topology, pdb.positions)
        ff = ForceField(*forcefield)

        modeller.addHydrogens(ff)
        modeller.addSolvent(
            ff,
            model=water_model,
            padding=padding_nm * unit.nanometer,
            ionicStrength=ionic_strength_M * unit.molar,
            neutralize=True,
        )

        system = ff.createSystem(
            modeller.topology, nonbondedMethod=PME, constraints=HBonds)
        integrator = LangevinMiddleIntegrator(
            temperature_K * unit.kelvin,
            1.0 / unit.picosecond,
            timestep_fs * unit.femtosecond,
        )
        integrator.setRandomNumberSeed(seed)

        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)

        if minimize:
            simulation.minimizeEnergy()

        simulation.context.setVelocitiesToTemperature(temperature_K * unit.kelvin, seed)
        n_steps = int(equil_ps * 1000 / timestep_fs)
        simulation.step(n_steps)

        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions().value_in_unit(unit.nanometer)

        # Select protein atoms: keep everything not in the water/ion set.
        protein_topology = Topology()
        protein_positions = []
        for chain in simulation.topology.chains():
            new_chain = None
            for residue in chain.residues():
                if residue.name.upper() in _SOLVENT_RESIDUE_NAMES:
                    continue
                if new_chain is None:
                    new_chain = protein_topology.addChain(chain.id)
                new_residue = protein_topology.addResidue(
                    residue.name, new_chain, residue.id)
                for atom in residue.atoms():
                    protein_topology.addAtom(atom.name, atom.element, new_residue)
                    protein_positions.append(Vec3(*positions[atom.index]))

        protein_positions = unit.Quantity(protein_positions, unit.nanometer)
        with open(pdb_out, "w") as fh:
            PDBFile.writeFile(protein_topology, protein_positions, fh)

    except RelaxError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raise as our typed error
        raise RelaxError(f"solvated relax failed: {exc}") from exc


def relax_in_water(
    pdb_in: str,
    pdb_out: str,
    *,
    forcefield: Sequence[str] = ("amber14-all.xml", "amber14/tip3pfb.xml"),
    water_model: str = "tip3p",
    padding_nm: float = 1.0,
    ionic_strength_M: float = 0.15,
    temperature_K: float = 300.0,
    timestep_fs: float = 2.0,
    equil_ps: float = 200.0,
    minimize: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """Relax ``pdb_in`` in explicit solvent, writing the protein-only result.

    Solvates the structure (AMBER14 + TIP3P-family water by default), briefly
    minimizes and equilibrates it under Langevin dynamics, then strips
    water/ions and writes the relaxed protein coordinates to ``pdb_out``.

    Returns a provenance dict recording every parameter plus the OpenMM
    version used, so callers/reports can cite exactly how the structure was
    produced. Raises ``RelaxError`` if the underlying OpenMM run fails.
    """
    import openmm

    provenance: Dict[str, Any] = {
        "forcefield": tuple(forcefield),
        "water_model": water_model,
        "padding_nm": padding_nm,
        "ionic_strength_M": ionic_strength_M,
        "temperature_K": temperature_K,
        "timestep_fs": timestep_fs,
        "equil_ps": equil_ps,
        "minimize": minimize,
        "seed": seed,
        "openmm_version": openmm.version.version,
    }

    _run_solvated_relax(
        pdb_in,
        pdb_out,
        forcefield=tuple(forcefield),
        water_model=water_model,
        padding_nm=padding_nm,
        ionic_strength_M=ionic_strength_M,
        temperature_K=temperature_K,
        timestep_fs=timestep_fs,
        equil_ps=equil_ps,
        minimize=minimize,
        seed=seed,
    )

    return provenance


class SolvatedRelaxStep(Step):
    """Relax a protein structure in explicit water/ions via ``relax_in_water``.

    Inputs
    ------
    structure_path : string
        Path to the input (unrelaxed) all-atom PDB.

    Outputs
    -------
    relaxed_path : string
        Path to the relaxed, protein-only PDB written by this step.
    """

    config_schema = {
        "forcefield": {"_type": "list[string]",
                        "_default": ["amber14-all.xml", "amber14/tip3pfb.xml"]},
        "water_model": {"_type": "string", "_default": "tip3p"},
        "padding_nm": {"_type": "float", "_default": 1.0},
        "ionic_strength_M": {"_type": "float", "_default": 0.15},
        "temperature_K": {"_type": "float", "_default": 300.0},
        "timestep_fs": {"_type": "float", "_default": 2.0},
        "equil_ps": {"_type": "float", "_default": 200.0},
        "minimize": {"_type": "boolean", "_default": True},
        "seed": {"_type": "integer", "_default": 0},
        "relaxed_path": {"_type": "string", "_default": ""},
    }

    def inputs(self):
        return {"structure_path": "string"}

    def outputs(self):
        return {"relaxed_path": "string"}

    def update(self, state, interval=None):
        structure_path = state["structure_path"]
        out_path = self.config["relaxed_path"] or (structure_path + ".relaxed.pdb")

        relax_in_water(
            structure_path,
            out_path,
            forcefield=tuple(self.config["forcefield"]),
            water_model=self.config["water_model"],
            padding_nm=float(self.config["padding_nm"]),
            ionic_strength_M=float(self.config["ionic_strength_M"]),
            temperature_K=float(self.config["temperature_K"]),
            timestep_fs=float(self.config["timestep_fs"]),
            equil_ps=float(self.config["equil_ps"]),
            minimize=bool(self.config["minimize"]),
            seed=int(self.config["seed"]),
        )

        return {"relaxed_path": out_path}
