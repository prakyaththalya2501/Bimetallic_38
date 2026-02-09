#!/usr/bin/env python3
import numpy as np
"""
GA + Basin Hopping (sequential refinement) for (bi)metallic clusters using ASE+EMT
with composition constraints and convergence plots.

Requirements:
    pip install ase numpy matplotlib

Usage:
    python ga_bh_comp_constrained.py

Notes:
- You can choose to run GA at a FIXED composition (all individuals share the same A_x B_(N-x))
  or with a SPECTRUM of compositions (each individual carries its own target x).
- Composition is enforced after crossover/mutation so it never drifts.
"""

import os, random, math
import matplotlib.pyplot as plt
from copy import deepcopy

from ase import Atoms
from ase.cluster import Octahedron
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.io import write
from ase.visualize import view
from mace.calculators import mace_mp

# ----------------------- User settings -----------------------
ELEMENTS = ['Cu', 'Ti']      # ELEMENTS[0] = A, ELEMENTS[1] = B
N_ATOMS =38                # total atoms in the cluster

# --- Composition control ---
COMPOSITION_MODE = "fixed"   # "fixed" or "spectrum"
TARGET_X = 28               # used only if COMPOSITION_MODE == "fixed"
# ----------------------------------------------------------------

# GA parameters
POP_SIZE = 15
N_GENERATIONS = 120
TOURNAMENT_SIZE = 5
ELITISM = 2
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.4
RATTLE_STD = 0.20   # Å for GA mutation displacement
RELAX_FMAX = 0.1   # eV/Å for local relaxation (BFGS)

# BH parameters (refines GA-best only)
BH_STEPS = 150
BH_TEMPERATURE_eV = 0.10
BH_STEP_STD = 0.25
BH_SWAP_PROB = 0.20  # swap attempt probability per BH trial (A<->B swap preserves counts)

OUT_DIR = "Cu3Ti_38"
SHOW_VIEW = False         # set True to open a viewer of the best structure at end
SEED = 42
# ---------------------------------------------------------------
calc = mace_mp(model="medium-mpa-0", device="cpu", dispersion=True, default_dtype="float64") #MACE Calculator Import
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

# ---------------- utilities ----------------
def make_seed_cluster(n_atoms, element=ELEMENTS[0]):
    """Make a compact truncated-octahedral-like seed and center it."""
    b = Octahedron(element, length=4, cutoff=1)
    pos = b.get_positions()
    center = pos.mean(axis=0)
    d2 = np.sum((pos - center)**2, axis=1)
    idx = np.argsort(d2)[:n_atoms]
    coords = pos[idx] - pos[idx].mean(axis=0)
    return coords

def recenter(atoms):
    p = atoms.get_positions()
    p -= p.mean(axis=0)
    atoms.set_positions(p)

def set_emt_and_relax(atoms):
    """Attach EMT, locally relax (BFGS), return energy (eV)."""
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    try:
        opt.run(fmax=RELAX_FMAX, steps=200)
    except Exception:
        pass
    return atoms.get_potential_energy()

def enforce_composition(atoms, nA, nB):
    """
    Ensure exactly nA of A (ELEMENTS[0]) and nB of B (ELEMENTS[1]).
    Converts excess atoms of a species into the other at random indices.
    """
    symbols = atoms.get_chemical_symbols()
    A, B = ELEMENTS[0], ELEMENTS[1]
    countA = symbols.count(A)
    countB = symbols.count(B)

    if countA + countB != len(symbols):
        # If other species are present (shouldn't happen with this script), coerce them into B first.
        others = [i for i, s in enumerate(symbols) if s not in (A, B)]
        for i in others:
            symbols[i] = B
        countA = symbols.count(A)
        countB = symbols.count(B)

    while countA > nA:
        idx = random.choice([i for i, s in enumerate(symbols) if s == A])
        symbols[idx] = B
        countA -= 1; countB += 1
    while countB > nB:
        idx = random.choice([i for i, s in enumerate(symbols) if s == B])
        symbols[idx] = A
        countB -= 1; countA += 1

    atoms.set_chemical_symbols(symbols)
    return atoms

def annotate_target_x(atoms, x):
    """Store target composition on the Atoms object."""
    if atoms.info is None:
        atoms.info = {}
    atoms.info['target_x'] = int(x)

def get_target_x(atoms):
    """Read target composition from Atoms.info; fallback to current A-count."""
    if atoms.info and 'target_x' in atoms.info:
        return int(atoms.info['target_x'])
    # fallback (shouldn't happen with this script)
    return atoms.get_chemical_symbols().count(ELEMENTS[0])

# ---------------- initialization ----------------
def make_individual_with_x(x):
    """Create A_x B_(N-x) individual; store target_x in atoms.info."""
    coords = make_seed_cluster(N_ATOMS, element=ELEMENTS[0])
    syms = [ELEMENTS[0]] * x + [ELEMENTS[1]] * (N_ATOMS - x)
    random.shuffle(syms)
    atoms = Atoms(symbols=syms, positions=coords)
    recenter(atoms)
    annotate_target_x(atoms, x)
    return atoms

def initialize_population():
    """
    If COMPOSITION_MODE == 'fixed': all individuals have A_TARGET_X B_(N - TARGET_X).
    If 'spectrum': create one individual for each x in [0..N_ATOMS], then sample/cycle to POP_SIZE.
    """
    if COMPOSITION_MODE.lower() == "fixed":
        x = int(TARGET_X)
        population = [make_individual_with_x(x) for _ in range(POP_SIZE)]
        return population

    # spectrum mode
    all_inds = [make_individual_with_x(x) for x in range(N_ATOMS + 1)]
    if POP_SIZE <= len(all_inds):
        return random.sample(all_inds, POP_SIZE)
    else:
        # cycle to reach POP_SIZE
        extra = [deepcopy(all_inds[i % len(all_inds)]) for i in range(POP_SIZE - len(all_inds))]
        return all_inds + extra

# ---------------- GA operators ----------------
def tournament(pop, energies, k=TOURNAMENT_SIZE):
    idx = random.sample(range(len(pop)), k)
    best = min(idx, key=lambda i: energies[i])
    return deepcopy(pop[best])

def plane_cut_and_splice(parent1, parent2):
    """
    Cut both parents by a random plane through their centers, combine halves.
    The child initially may have wrong composition; we'll enforce it later.
    The child inherits a target_x (composition) from one of the parents randomly.
    """
    p1, p2 = parent1.copy(), parent2.copy()
    pos1 = p1.get_positions() - p1.get_positions().mean(axis=0)
    pos2 = p2.get_positions() - p2.get_positions().mean(axis=0)
    s1, s2 = p1.get_chemical_symbols(), p2.get_chemical_symbols()

    n = np.random.randn(3)
    n /= np.linalg.norm(n)
    mask1 = (pos1 @ n) > 0
    mask2 = (pos2 @ n) <= 0

    coords, syms = [], []
    for i, m in enumerate(mask1):
        if m:
            coords.append(pos1[i]); syms.append(s1[i])
    for j, m in enumerate(mask2):
        if m:
            coords.append(pos2[j]); syms.append(s2[j])

    # adjust size
    while len(coords) < N_ATOMS:
        pick = random.randrange(len(p1))
        coords.append(pos1[pick] + np.random.randn(3)*0.01)
        syms.append(s1[pick])
    if len(coords) > N_ATOMS:
        keep = random.sample(range(len(coords)), N_ATOMS)
        coords = [coords[i] for i in keep]
        syms = [syms[i] for i in keep]

    child = Atoms(symbols=syms, positions=np.array(coords))
    recenter(child)

    # inherit or set target_x
    if COMPOSITION_MODE.lower() == "fixed":
        tx = int(TARGET_X)
    else:
        tx = random.choice([get_target_x(parent1), get_target_x(parent2)])
    annotate_target_x(child, tx)

    # enforce composition
    child = enforce_composition(child, nA=tx, nB=N_ATOMS - tx)
    return child

def mutate_rattle_and_swap(atoms, rattle_std=RATTLE_STD, swap_prob=0.2):
    """
    Rattle positions; optionally perform a species swap between different types.
    After mutation, enforce composition to keep counts fixed.
    """
    pos = atoms.get_positions().copy()
    pos += np.random.normal(scale=rattle_std, size=pos.shape)
    atoms.set_positions(pos)

    # optional species swap that preserves counts
    if random.random() < swap_prob and len(ELEMENTS) > 1:
        s = atoms.get_chemical_symbols()
        A, B = ELEMENTS[0], ELEMENTS[1]
        idxA = [i for i, si in enumerate(s) if si == A]
        idxB = [i for i, si in enumerate(s) if si == B]
        if idxA and idxB:
            i = random.choice(idxA); j = random.choice(idxB)
            s[i], s[j] = s[j], s[i]
            atoms.set_chemical_symbols(s)

    recenter(atoms)

    # enforce composition (critical)
    tx = get_target_x(atoms)
    atoms = enforce_composition(atoms, nA=tx, nB=N_ATOMS - tx)
    return atoms

# ---------------- GA main ----------------
def run_ga():
    population = initialize_population()
    # initial local relax
    energies = [set_emt_and_relax(ind) for ind in population]

    best_overall, best_energy = None, float('inf')
    best_history = []  # best energy per generation (for plotting)

    for gen in range(1, N_GENERATIONS + 1):
        new_pop, new_energies = [], []

        # Elitism
        sorted_idx = sorted(range(len(population)), key=lambda i: energies[i])
        elites = [deepcopy(population[i]) for i in sorted_idx[:ELITISM]]
        for e in elites:
            # ensure composition (defensive)
            tx = get_target_x(e)
            e = enforce_composition(e, nA=tx, nB=N_ATOMS - tx)
            eE = set_emt_and_relax(e)
            new_pop.append(e); new_energies.append(eE)

        # Fill the rest
        while len(new_pop) < POP_SIZE:
            parent1 = tournament(population, energies)
            parent2 = tournament(population, energies)
            if random.random() < CROSSOVER_RATE:
                child = plane_cut_and_splice(parent1, parent2)
            else:
                child = deepcopy(parent1)

            if random.random() < MUTATION_RATE:
                child = mutate_rattle_and_swap(child)

            # final enforcement before eval (safety)
            tx = get_target_x(child)
            child = enforce_composition(child, nA=tx, nB=N_ATOMS - tx)

            e_child = set_emt_and_relax(child)
            new_pop.append(child); new_energies.append(e_child)

        population, energies = new_pop, new_energies

        gen_best_idx = int(np.argmin(energies))
        gen_best = population[gen_best_idx]
        gen_best_energy = energies[gen_best_idx]

        if gen_best_energy < best_energy:
            best_energy = gen_best_energy
            best_overall = deepcopy(gen_best)
            write(os.path.join(OUT_DIR, f"best_gen{gen:03d}.xyz"), best_overall)

        best_history.append(best_energy)
        tx = get_target_x(gen_best)
        print(f"Gen {gen:3d} | Best E = {gen_best_energy:.6f} eV | Overall best = {best_energy:.6f} eV | target_x = {tx}")

    # Plot GA convergence
    plt.figure()
    plt.plot(range(1, N_GENERATIONS + 1), best_history, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Best Energy (eV)")
    plt.title("GA Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ga_convergence.png"), dpi=300)
    plt.close()

    return best_overall, best_energy, best_history

# ---------------- Basin Hopping (sequential refinement) ----------------
def bh_trial_move(atoms, step_std, swap_prob):
    """Propose a new structure: rattle positions + optional A<->B swap (counts preserved)."""
    trial = atoms.copy()
    trial = mutate_rattle_and_swap(trial, rattle_std=step_std, swap_prob=swap_prob)
    # enforce composition (defensive)
    tx = get_target_x(trial)
    trial = enforce_composition(trial, nA=tx, nB=N_ATOMS - tx)
    return trial

def bh_refine(start_atoms, steps=BH_STEPS):
    """
    Basin Hopping refinement on positions with composition fixed.
    Metropolis acceptance at BH_TEMPERATURE_eV; local relax via BFGS.
    """
    current = deepcopy(start_atoms)
    tx = get_target_x(current)
    current = enforce_composition(current, nA=tx, nB=N_ATOMS - tx)
    Ec = set_emt_and_relax(current)

    best, Ebest = deepcopy(current), Ec
    history = [Ec]

    for step in range(1, steps + 1):
        trial = bh_trial_move(current, BH_STEP_STD, BH_SWAP_PROB)
        Et = set_emt_and_relax(trial)
        dE = Et - Ec
        accept = dE <= 0.0 or random.random() < math.exp(-dE / BH_TEMPERATURE_eV)

        if accept:
            current, Ec = trial, Et
            if Ec < Ebest:
                best, Ebest = deepcopy(current), Ec
                write(os.path.join(OUT_DIR, f"bh_best_step{step:04d}.xyz"), best)

        history.append(Ec)

        if step % 20 == 0:
            print(f" BH step {step:4d}: Ec = {Ec:.6f} eV | best = {Ebest:.6f} eV | target_x = {tx}")

    # Plot BH trajectory
    plt.figure()
    plt.plot(range(len(history)), history, marker='.')
    plt.xlabel("BH Step")
    plt.ylabel("Energy (eV)")
    plt.title("Basin Hopping Refinement")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bh_trajectory.png"), dpi=300)
    plt.close()

    return best, Ebest, history

# ---------------- Orchestrator ----------------
def run_ga_then_bh():
    ga_best, ga_Ebest, ga_hist = run_ga()
    write(os.path.join(OUT_DIR, "ga_best.xyz"), ga_best)
    print(f"GA finished. Best energy: {ga_Ebest:.6f} eV | target_x = {get_target_x(ga_best)}")

    bh_best, bh_Ebest, bh_hist = bh_refine(ga_best)
    write(os.path.join(OUT_DIR, "bh_best.xyz"), bh_best)
    print(f"BH refinement finished. Best energy: {bh_Ebest:.6f} eV")

    if bh_Ebest < ga_Ebest:
        overall_best, overall_E = bh_best, bh_Ebest
        tag = "BH-refined"
    else:
        overall_best, overall_E = ga_best, ga_Ebest
        tag = "GA-best"

    write(os.path.join(OUT_DIR, "best_overall.xsd"), overall_best)
    print(f"Overall best ({tag}) = {overall_E:.6f} eV | target_x = {get_target_x(overall_best)}")

    if SHOW_VIEW:
        view(overall_best)
    return overall_best, overall_E, ga_hist, bh_hist

# ---------------- main ----------------
if __name__ == "__main__":
    best, Ebest, ga_hist, bh_hist = run_ga_then_bh()
