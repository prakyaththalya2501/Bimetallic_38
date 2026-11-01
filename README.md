# Machine leaning-aided Screening of Functional and Sustainable Bimetallic Catalysts for Dry Methane Reforming

This repository contains the **Genetic Algorithm (GA)–MACE optimization workflow** and the corresponding computational data for studying catalytic reactions on bimetallic nanoparticles.
## Repository Contents
### 1. Genetic Algorithm (GA) Code
Contains the source code used for genetic algorithm-based generation and optimization of bimetallic nanoparticle structures.
### 2. GA-MACE Optimized Structures
Contains the optimized bimetallic nanoparticle configurations obtained through the GA-MACE workflow.
### 3. MACE-Generated Reactant and Product Structures
Contains the reactant and product POSCAR files generated using MACE for the identified optimum adsorption/binding sites on the bimetallic nanoparticles.
### 4. VASP-Optimized Structures
Contains the DFT-relaxed structures obtained using VASP, starting from the optimum binding-site configurations identified through MACE.
### 5. Transition State Structures (`TS_CONTCARs/`)
Contains the transition-state CONTCAR files obtained from VASP calculations, organized according to the respective bimetallic compositions.
## Computational Workflow
1. Generation and optimization of bimetallic nanoparticles using Genetic Algorithms and MACE.
2. Identification of optimum reactant and product binding sites using MACE.
3. DFT optimization and validation of selected configurations using VASP.
4. Transition-state calculations to investigate the corresponding catalytic reaction pathways.

---

**Note:** Optimized structures and calculation outputs are organized according to their respective bimetallic compositions wherever applicable.
