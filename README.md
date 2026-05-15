# Semilocal Exchange Potential: SCAN and r²SCAN

This repository contains the data and Python scripts for computing and visualizing the **semilocal part of the exchange potential** (`v_x^sl`) for SCAN and r²SCAN meta-GGA functionals, evaluated on Hartree–Fock atomic orbitals for **H**, **Si**, and **Xe**.

## Background

The exchange-correlation potential in DFT meta-GGA functionals depends on the kinetic energy density `τ`. The **semilocal exchange potential** is defined via:

```
v_x^sl(r) = ∂(n ε_x)/∂n − ∇·[∂(n ε_x)/∂(∇n)]
```

This project computes `v_x^sl` using:
- **SCAN** — Strongly Constrained and Appropriately Normed functional
- **r²SCAN** — Regularized-restored SCAN (default `α̅` denominator)
- **r²SCAN@LKT** — r²SCAN with the Laplacian Kinetic energy density (LKT) regularization
- **r²SCAN@PGS** — r²SCAN with the PGS iso-orbital indicator

Input densities are constructed from **Hartree–Fock Slater-type / numerical orbitals** (Koga tabulations).

---

## Repository Structure

```
.
├── H/                         # Hydrogen atom
│   ├── hf_orbitals.py         # Build HF density, gradient, τ from Koga data
│   ├── plot.py                # Generate v_x^sl comparison figure
│   ├── H_Koga_radial_data.dat # Radial density data
│   ├── H_Koga_orbitals.dat    # Orbital coefficients
│   ├── R2SCAN/
│   │   ├── vx_r2scan.py       # Compute v_x^sl for r²SCAN (+ LKT & PGS variants)
│   │   ├── r2scan.dat         # Output: r²SCAN semilocal exchange potential
│   │   ├── r2scan_LKT.dat     # Output: r²SCAN@LKT variant
│   │   └── r2scan_PGS.dat     # Output: r²SCAN@PGS variant
│   └── SCAN/
│       ├── vx_scan.py         # Compute v_x^sl for SCAN
│       └── scan.dat           # Output: SCAN semilocal exchange potential
│
├── Si/                        # Silicon atom (14 electrons)
│   ├── hf_orbitals.py
│   ├── plot.py
│   ├── Si_Koga_radial_data.dat
│   ├── Si_Koga_orbitals.dat
│   ├── R2SCAN/  ...
│   └── SCAN/    ...
│
├── Xe/                        # Xenon atom (54 electrons)
│   ├── hf_orbitals.py
│   ├── plot.py
│   ├── Xe_Koga_radial_data.dat
│   ├── R2SCAN/  ...
│   └── SCAN/    ...
│
└── assets/                    # Figures for the GitHub Pages site
    ├── H_vx_r2scan_scan.pdf
    ├── Si_vx_r2scan_scan.pdf
    └── Xe_vx_r2scan_scan.pdf
```

---

## How to Run

### Requirements

```bash
pip install numpy matplotlib
```

### Reproduce results for a given element (e.g. Hydrogen)

```bash
cd H/R2SCAN
python vx_r2scan.py        # computes r2scan.dat, r2scan_LKT.dat, r2scan_PGS.dat

cd ../SCAN
python vx_scan.py          # computes scan.dat

cd ..
python plot.py             # generates the comparison figure
```

Repeat analogously for `Si/` and `Xe/`.

---

## Key Physics

| Quantity | Symbol | Description |
|----------|--------|-------------|
| Reduced gradient | `s` | `\|∇n\| / (2kF n)` |
| Iso-orbital indicator | `α̅` | `(τ - τ_W) / τ_denom` |
| Enhancement factor | `Fx(s, α̅)` | Functional-specific boost over LDA |
| LDA exchange energy density | `ε_x^LDA` | `−CX · n^{1/3}` |
| Semilocal exchange potential | `v_x^sl` | Functional derivative acting on radial density |

The Lagrange derivative method (Seino & Nakai) is used for numerical radial derivatives on the non-uniform Koga grid.

---

## Data Format

All `.dat` files are plain-text, two-column (space-separated):

```
# r  vx
0.001   -12.345
0.002   -11.980
...
```

Units: **r** in Bohr, **v_x^sl** in Hartree.

---

## License

Data and scripts are released for academic use. If you use this work, please cite appropriately.
