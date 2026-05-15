"""
Si_Koga_density.py
==================
Hartree-Fock density of Silicon (Z=14) using the Koga 1999 STO basis.

Reference:
    T. Koga et al., Int. J. Quantum Chem. 71, 491 (1999).

Koga 1999 data for Si:
    E = -288.854362454 Ha
    T =  288.854362538 Ha    V = -577.708724992 Ha

Configuration: [Ne] 3s² 3p²
    1s(2) 2s(2) 2p(6) 3s(2) 3p(2)   → 14 electrons  (Z = 14)

ORBITAL STRUCTURE
-----------------
S orbitals  (l=0):  1s (occ=2),  2s (occ=2),  3s (occ=2)
P orbitals  (l=1):  2p (occ=6),  3p (occ=2)      ← 3p is HALF-FILLED

NOTE: 2p is fully filled (occ=6) but 3p has only 2 electrons.
They use the SAME set of 10 primitive Gaussian exponents/types.
"""

import numpy as np
from math import factorial, sqrt, pi
import matplotlib.pyplot as plt


# ============================================================================
# 1.  STO basis — Koga 1999 coefficients for Si
# ============================================================================

# ── S orbitals (l=0) ─────────────────────────────────────────────────────────
Si_1S = [
    (1, 40.744005,  0.0027330), (1, 32.345799, -0.0101378),
    (1, 14.154495, -0.9311367), (2, 12.216360, -0.0745227),
    (1,  6.257945, -0.0042718), (2,  4.879641, -0.0034503),
    (1,  3.586780,  0.0037882), (1,  2.070236, -0.0003537),
    (1,  0.975003,  0.0001180), (1,  0.770863, -0.0000551),
]
Si_2S = [
    (1, 40.744005,  0.0010884), (1, 32.345799, -0.0032055),
    (1, 14.154495, -0.2917220), (2, 12.216360, -0.1774382),
    (1,  6.257945, -0.3282571), (2,  4.879641,  0.3203634),
    (1,  3.586780,  1.0983224), (1,  2.070236,  0.0013121),
    (1,  0.975003,  0.0012855), (1,  0.770863, -0.0003025),
]
Si_3S = [
    (1, 40.744005, -0.0003241), (1, 32.345799,  0.0009458),
    (1, 14.154495,  0.0659012), (2, 12.216360,  0.0416884),
    (1,  6.257945,  0.6266028), (2,  4.879641,  0.5323327),
    (1,  3.586780, -1.3386717), (1,  2.070236, -0.8164418),
    (1,  0.975003,  1.6628928), (1,  0.770863, -0.0327784),
]

# ── P orbitals (l=1) — shared 10-primitive basis ─────────────────────────────
# IMPORTANT: first exponent is 35.986281 (not 32.986281)
Si_2P = [
    (2, 35.986281, -0.0000771), (2, 20.811651,  0.0019986),
    (2, 11.079502,  0.0027286), (2,  8.569582,  0.1324846),
    (2,  5.733707,  0.2686232), (2,  4.091940,  0.5215984),
    (2,  3.057286,  0.1236799), (2,  1.252393,  0.0026814),
    (2,  0.870155, -0.0007868), (3,  0.770863,  0.0002433),
]
Si_3P = [
    (2, 35.986281,  0.0000186), (2, 20.811651, -0.0004618),  # coeff: 0.0000186
    (2, 11.079502,  0.0010049), (2,  8.569582, -0.0298094),
    (2,  5.733707, -0.0732137), (2,  4.091940, -0.0253179),
    (2,  3.057286, -0.2177402), (2,  1.252393,  0.3957806),  # coeff: -0.2177402
    (2,  0.870155,  0.6899150), (3,  0.770863,  0.0234931),
]

# Occupations — Si: 1s²2s²2p⁶3s²3p²
OCC_1S = 2;  OCC_2S = 2;  OCC_3S = 2
OCC_2P = 6;  OCC_3P = 2   # 3p is half-filled: only 2 electrons


# ============================================================================
# 2.  Radial grid
# ============================================================================
def make_log_grid(r0=1e-6, rmax=40.0, nmax=4000):
    h  = np.log(rmax / r0) / (nmax - 1)
    al = np.exp(h)
    r  = r0 * al ** np.arange(nmax)
    return r, al, h


# ============================================================================
# 3.  Simpson integrator on log mesh
# ============================================================================
def fmoment(moment, fnorm, r, fr):
    al   = np.log(r[1] / r[0])
    fm   = fr * r ** (moment + 3)
    mmax = len(fm)
    if mmax % 2 == 0:
        mmax -= 1;  fm = fm[:mmax]
    imask    = np.arange(mmax) % 2
    esum     = np.add.reduce(imask * fm)
    antimask = 1 - imask[1:-1]
    osum     = np.add.reduce(antimask * fm[1:-1])
    fmom     = al * (4.0*esum + 2.0*osum + fm[0] + fm[-1]) / 3.0
    return fmom / fnorm


# ============================================================================
# 4.  STO evaluation
# ============================================================================
def sto_norm(n: int, zeta: float) -> float:
    return sqrt((2.0 * zeta) ** (2 * n + 1) / factorial(2 * n))

def eval_sto(params, r: np.ndarray):
    R, dR, d2R = np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)
    for (n, zeta, c) in params:
        N_  = sto_norm(n, zeta)
        G   = N_ * r ** (n - 1) * np.exp(-zeta * r)
        dG  = ((n - 1) / r - zeta) * G
        d2G = ((n - 1) / r - zeta) ** 2 * G - (n - 1) / r ** 2 * G
        R   += c * G;  dR += c * dG;  d2R += c * d2G
    return R, dR, d2R


# ============================================================================
# 5.  Build grid and evaluate all orbitals
# ============================================================================
r, al_ratio, h_grid = make_log_grid(r0=1e-6, rmax=40.0, nmax=4000)

R1s, dR1s, d2R1s = eval_sto(Si_1S, r)
R2s, dR2s, d2R2s = eval_sto(Si_2S, r)
R3s, dR3s, d2R3s = eval_sto(Si_3S, r)
R2p, dR2p, d2R2p = eval_sto(Si_2P, r)
R3p, dR3p, d2R3p = eval_sto(Si_3P, r)


# ============================================================================
# 6.  Density and its derivatives
# ============================================================================
n = (
    OCC_1S * R1s**2 + OCC_2S * R2s**2 + OCC_3S * R3s**2
  + OCC_2P * R2p**2 + OCC_3P * R3p**2
) / (4.0 * pi)

gradn = (
    OCC_1S * 2.0 * R1s*dR1s + OCC_2S * 2.0 * R2s*dR2s + OCC_3S * 2.0 * R3s*dR3s
  + OCC_2P * 2.0 * R2p*dR2p + OCC_3P * 2.0 * R3p*dR3p
) / (4.0 * pi)

gradn2  = gradn ** 2

d2n_dr2 = (
    OCC_1S * 2.0 * (dR1s**2 + R1s*d2R1s)
  + OCC_2S * 2.0 * (dR2s**2 + R2s*d2R2s)
  + OCC_3S * 2.0 * (dR3s**2 + R3s*d2R3s)
  + OCC_2P * 2.0 * (dR2p**2 + R2p*d2R2p)
  + OCC_3P * 2.0 * (dR3p**2 + R3p*d2R3p)
) / (4.0 * pi)

lapl = d2n_dr2 + (2.0 / r) * gradn


# ============================================================================
# 7.  Kinetic energy densities
# ============================================================================
tau_ks = (
    OCC_1S * dR1s**2 + OCC_2S * dR2s**2 + OCC_3S * dR3s**2
  + OCC_2P * (dR2p**2 + 2.0/r**2 * R2p**2)
  + OCC_3P * (dR3p**2 + 2.0/r**2 * R3p**2)
) / (8.0 * pi)

n_safe = np.maximum(n, 1e-30)
tau_vw = gradn2 / (8.0 * n_safe)
C_TF   = (3.0 / 10.0) * (3.0 * pi**2) ** (2.0 / 3.0)
tau_tf = C_TF * n_safe ** (5.0 / 3.0)


# ============================================================================
# 8.  meta-GGA reduced variables
# ============================================================================
kF    = np.maximum((3.0 * pi**2 * n_safe) ** (1.0/3.0), 1e-12)
s_sqr = gradn2 / (4.0 * kF**2 * n_safe**2)
q     = lapl   / (4.0 * (3.0 * pi**2)**(2.0/3.0) * n_safe**(5.0/3.0))


# ============================================================================
# 9.  Sanity checks
# ============================================================================
N_elec = fmoment(0, 1.0/(4.0*pi), r, n)
T_KS   = fmoment(0, 1.0/(4.0*pi), r, tau_ks)
T_vW   = fmoment(0, 1.0/(4.0*pi), r, tau_vw)
pauli  = tau_ks - tau_vw

print()
print("=" * 56)
print("Si  Z=14  Koga 1999 HF-STO")
print("Config: 1s² 2s² 2p⁶ 3s² 3p²")
print("Grid: r0=1e-6, rmax=40 Bohr, N=4000")
print("=" * 56)
print(f"Electrons  : {N_elec:.8f}     (exact = 14)")
print(f"T_KS       : {T_KS:.8f}  (Koga  = 288.854362538)")
print(f"T_vW       : {T_vW:.8f}")
print(f"Pauli KED  : min = {pauli.min():.2e}   (≥ 0 required)")
print(f"             max = {pauli.max():.2e}")


# ============================================================================
# 10.  Per-orbital electron count check
# ============================================================================
print()
print("Per-orbital electron counts:")
for label, params, occ in [
    ("1s", Si_1S, OCC_1S),
    ("2s", Si_2S, OCC_2S),
    ("3s", Si_3S, OCC_3S),
    ("2p", Si_2P, OCC_2P),
    ("3p", Si_3P, OCC_3P),
]:
    R_, _, _ = eval_sto(params, r)
    ni = occ * R_**2 / (4.0*pi)
    ne = fmoment(0, 1.0/(4.0*pi), r, ni)
    print(f"  {label} (occ={occ}): {ne:.8f}   (exact = {occ})")


# ============================================================================
# 11.  Save
# ============================================================================
np.savetxt(
    "Si_Koga_radial_data.dat",
    np.column_stack((r, n, gradn, gradn2, lapl, tau_ks, tau_tf, s_sqr, q, tau_vw)),
    header=(
        "r   n   gradn   gradn2   lapl   tau_KS   tauTF   s2   q   tau_vW\n"
        f"Si Z=14  Koga1999  E=-288.854362454 Ha  T_KS={T_KS:.8f} Ha"
    )
)

np.savetxt(
    "Si_Koga_orbitals.dat",
    np.column_stack((r,
        R1s, dR1s,
        R2s, dR2s,
        R3s, dR3s,
        R2p, dR2p,
        R3p, dR3p)),
    header=(
        "r  R1s dR1s  R2s dR2s  R3s dR3s  R2p dR2p  R3p dR3p\n"
        "Si Koga 1999 per-orbital radial wavefunctions and derivatives.\n"
        "Occupations: 1s=2, 2s=2, 3s=2, 2p=6, 3p=2\n"
        "Column index: 0=r, 1-2=R1s/dR1s, 3-4=R2s/dR2s, 5-6=R3s/dR3s,"
        " 7-8=R2p/dR2p, 9-10=R3p/dR3p"
    )
)
print()
print("Saved: Si_Koga_radial_data.dat  (10 cols)")
print("Saved: Si_Koga_orbitals.dat     (11 cols)")


# ============================================================================
# 12.  Plots
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Si (Z=14) — Koga 1999 Hartree-Fock STO", fontsize=14)

pairs = [
    (n,      r"$n(r)$",                True),
    (gradn2, r"$|\nabla n|^2$",        True),
    (lapl,   r"$\nabla^2 n$",          False),
    (tau_ks, r"$\tau_{KS}$",           True),
    (tau_tf, r"$\tau_{TF}$",           True),
    (s_sqr,  r"$s^2$ (reduced grad²)", True),
]

for ax, (y, lbl, logy) in zip(axes.flat, pairs):
    ax.plot(r, y, lw=1.6, color='steelblue')
    ax.set_xscale("log")
    if logy:
        pos = y[y > 0]
        if len(pos):
            ax.set_yscale("log")
    ax.set_xlabel("r  (Bohr)")
    ax.set_ylabel(lbl)
    ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("Si_Koga_plots.png", dpi=150)
plt.show()
print("Saved: Si_Koga_plots.png")
