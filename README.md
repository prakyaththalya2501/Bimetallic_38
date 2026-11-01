# Machine leaning-aided Screening of Functional and Sustainable Bimetallic Catalysts for Dry Methane Reforming

This repository contains the Genetic Algorithm (GA)–MACE optimization workflow, optimized bimetallic nanoparticle structures, and the corresponding DFT calculations performed using VASP.

Repository Contents

Genetic Algorithm (GA) Code
Source code used for the genetic algorithm-based generation and optimization of bimetallic nanoparticle structures.

GA-MACE Optimized Structures
Optimized bimetallic nanoparticle configurations obtained using the GA-MACE workflow.

MACE-Generated Reactant and Product Structures
Reactant and product POSCAR structures generated using MACE, corresponding to the identified optimum adsorption/binding sites on the bimetallic nanoparticles.

VASP-Optimized Structures
DFT-relaxed structures obtained using VASP, starting from the optimum binding-site configurations identified through MACE.

Transition State Structures (TS_CONTCARs/)
VASP-generated transition-state CONTCAR files, organized according to the respective bimetallic compositions.

Workflow Overview

The repository follows a computational workflow involving GA-based nanoparticle optimization, MACE-assisted identification of optimum binding sites, DFT validation using VASP, and transition-state calculations for catalytic reaction studies.
