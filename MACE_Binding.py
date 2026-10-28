#!/usr/bin/env python3

# ============================================================
# Physically consistent CH4 → CH3 + H dissociation on clusters
# ============================================================

from ase.cluster import FaceCenteredCubic
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.build import molecule
from ase import Atoms
from ase.io import write, read
from ase.io.vasp import write_vasp
from ase.optimize.sciopt import SciPyFminCG
from mace.calculators import mace_mp

import numpy as np
import os
import glob
import pandas as pd

# ================= MACE Calculator =================
calc = mace_mp(
    model="medium-mpa-0",
    device="cpu",
    dispersion=True,
    default_dtype="float64"
)

# ================= Input directory =================
xsd_dir = "./collected_xsd"        # directory containing nanoparticle .xsd files
clusters = sorted(glob.glob(os.path.join(xsd_dir, "*.xsd")))

# ===================================================
# Utility: surface normal (radial)
# ===================================================
def get_surface_normal(cluster, idx):
    center = cluster.get_center_of_mass()
    v = cluster[idx].position - center
    return v / np.linalg.norm(v)

# ===================================================
# Main loop over nanoparticles
# ===================================================
for n_idx, cluster_file in enumerate(clusters, start=1):

    print(f"\nProcessing cluster: {cluster_file}")
    np_cluster = read(cluster_file)

    # ---------- Identify surface atoms ----------
    cutoffs = natural_cutoffs(np_cluster, mult=1.0)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(np_cluster)

    surface_atoms = []
    for i, atom in enumerate(np_cluster):
        neigh, _ = nl.get_neighbors(i)
        if len(neigh) < 10:
            surface_atoms.append(i)

    print(f"Identified {len(surface_atoms)} surface atoms")

    # ---------- Output containers ----------
    reaction_energies = []
    energies = []
    structures = []
    cluster_name = os.path.splitext(os.path.basename(cluster_file))[0]
    out_dir = os.path.join("OUTPUT", cluster_name)
    os.makedirs(out_dir, exist_ok=True)

    # ===================================================
    # Loop over surface sites
    # ===================================================
    for site in surface_atoms:

        base_pos = np_cluster[site].position
        normal = get_surface_normal(np_cluster, site)

        # ===================================================
        # 1️⃣ Build CH4 with ONE H pointing toward surface
        # ===================================================
        ch4 = molecule("CH4")

        C_idx = 0
        H_indices = [i for i, a in enumerate(ch4) if a.symbol == "H"]

        # choose hydrogen to point to surface
        H_target = H_indices[0]

        v_CH = ch4[H_target].position - ch4[C_idx].position
        v_CH /= np.linalg.norm(v_CH)

        rot_axis = np.cross(v_CH, normal)
        if np.linalg.norm(rot_axis) > 1e-6:
            rot_axis /= np.linalg.norm(rot_axis)
            angle = np.degrees(np.arccos(np.dot(v_CH, normal)))
            ch4.rotate(angle, v=rot_axis, center=ch4[C_idx].position)

        ch4.translate(base_pos + 1.8 * normal - ch4[C_idx].position)

        ads_CH4 = np_cluster.copy() + ch4

        # ===================================================
        # 2️⃣ Identify nearest hydrogen to surface atom
        # ===================================================
        dists = []
        for h in H_indices:
            abs_idx = len(np_cluster) + h
            d = np.linalg.norm(ads_CH4[abs_idx].position - base_pos)
            dists.append((h, d))

        diss_H, _ = min(dists, key=lambda x: x[1])
        abs_H_idx = len(np_cluster) + diss_H

        # ===================================================
        # 3️⃣ Create CH3 by deleting THAT hydrogen
        # ===================================================
        ads_CH3_H = ads_CH4.copy()
        del ads_CH3_H[abs_H_idx]

        # ===================================================
        # 4️⃣ Place dissociated H on surface (bridge preferred)
        # ===================================================
        neigh_i, _ = nl.get_neighbors(site)
        surf_neigh = [j for j in neigh_i if j in surface_atoms]

        if surf_neigh:
            j = surf_neigh[0]
            mid = 0.5 * (np_cluster[site].position +
                         np_cluster[j].position)
            lift = mid - np_cluster.get_center_of_mass()
            lift /= np.linalg.norm(lift)
            h_pos = mid + 1.2 * lift
        else:
            h_pos = base_pos + 1.5 * normal

        ads_CH3_H += Atoms("H", positions=[h_pos])

        # ===================================================
        # 5️⃣ Relax structures
        # ===================================================
        ads_CH4.calc = calc
        dyn_r = SciPyFminCG(ads_CH4, logfile=None)
        dyn_r.run(fmax=0.1)
        e_ch4 = ads_CH4.get_potential_energy()

        ads_CH3_H.calc = calc
        dyn_p = SciPyFminCG(ads_CH3_H, logfile=None)
        dyn_p.run(fmax=0.1)
        e_ch3_h = ads_CH3_H.get_potential_energy()

        # ===================================================
        # 6️⃣ Save structures
        # ===================================================
        write(f"{out_dir}/CH4_site_{site}.xsd", ads_CH4)
        write(f"{out_dir}/CH3_H_site_{site}.xsd", ads_CH3_H)

        delta_e = e_ch3_h - e_ch4
        reaction_energies.append((site, delta_e))
        energies.append((site, e_ch3_h, e_ch4))
        structures.append((ads_CH3_H, ads_CH4))

        print(f"Site {site}: ΔE = {delta_e:.3f} eV")

    # ===================================================
    # 7️⃣ Lowest-energy site
    # ===================================================
    min_site, min_de = min(reaction_energies, key=lambda x: x[1])
    idx = [s for s, _ in reaction_energies].index(min_site)

    prod, reac = structures[idx]

    os.makedirs(f"OUTPUT_POSCARS/{cluster_name}/Product", exist_ok=True)
    os.makedirs(f"OUTPUT_POSCARS/{cluster_name}/Reactant", exist_ok=True)

    write_vasp(
        f"OUTPUT_POSCARS/{cluster_name}/Product/POSCAR",
        prod, direct=True, vasp5=True, sort=True
    )
    write_vasp(
        f"OUTPUT_POSCARS/{cluster_name}/Reactant/POSCAR",
        reac, direct=True, vasp5=True, sort=True
    )

    # ===================================================
    # 8️⃣ Save CSV
    # ===================================================
    df_e = pd.DataFrame(
        energies, columns=["Site Index", "E_CH3+H (eV)", "E_CH4 (eV)"]
    )
    df_d = pd.DataFrame(
        reaction_energies, columns=["Site Index", "ΔE_rxn (eV)"]
    )
    df = pd.merge(df_e, df_d, on="Site Index")
    df.to_csv(f"{out_dir}/Combined_Energy_Table.csv", index=False)

    print(f"Lowest ΔE: site {min_site}, ΔE = {min_de:.3f} eV")
