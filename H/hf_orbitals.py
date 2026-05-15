"""
H_Koga_density.py
=================
Hartree-Fock density of Hydrogen (Z=1) using the Koga 1999 STO basis.

Reference:
    T. Koga et al., Int. J. Quantum Chem. 71, 491 (1999).

Koga 1999 data for H:
    E = -0.50000000 Ha
    T =  0.50000000 Ha    V = -1.00000000 Ha

Configuration: 1s¹   → 1 electron  (Z = 1)

ORBITAL STRUCTURE
-----------------
S orbitals (l=0):  1s (occ=1)  — single primitive STO, EXACT solution

NOTE:
  H is the only atom where the HF basis is a single Slater function
  1S with zeta=1.0, coefficient=1.0 — this is the exact HF ground state.
  Consequently:
    alpha_bar = (tau_KS - tau_vW) / (tau_TF + eta*tau_vW) = 0  EXACTLY
  because for a single orbital tau_KS = tau_vW identically.
  The Pauli kinetic energy is ZERO everywhere (not just numerically small).
"""

import numpy as np
from math import factorial, sqrt, pi
import matplotlib.pyplot as plt


# ============================================================================
# 1.  STO basis — single primitive, exact HF solution
# ============================================================================
H_1S = [
    (1, 1.000000, 1.0000000),   # (n, zeta, coefficient)
]

OCC_1S = 1   # hydrogen: 1 electron in 1s


# ============================================================================
# 2.  Radial grid  (rmax=20 Bohr sufficient for H)
# ============================================================================
def make_log_grid(r0=1e-6, rmax=20.0, nmax=2000):
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
# 5.  Build grid and evaluate orbital
# ============================================================================
r, al_ratio, h_grid = make_log_grid(r0=1e-6, rmax=20.0, nmax=2000)

R1s, dR1s, d2R1s = eval_sto(H_1S, r)


# ============================================================================
# 6.  Density and its derivatives  (single orbital)
# ============================================================================
n       = OCC_1S * R1s**2 / (4.0 * pi)
gradn   = OCC_1S * 2.0 * R1s * dR1s / (4.0 * pi)
gradn2  = gradn ** 2
d2n_dr2 = OCC_1S * 2.0 * (dR1s**2 + R1s*d2R1s) / (4.0 * pi)
lapl    = d2n_dr2 + (2.0 / r) * gradn


# ============================================================================
# 7.  Kinetic energy densities
#     H 1s: l=0 → no centrifugal term
#     Single orbital → tau_KS = tau_vW exactly → Pauli = 0 exactly
# ============================================================================
tau_ks = OCC_1S * dR1s**2 / (8.0 * pi)

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
print("H  Z=1  Koga 1999 HF-STO  (exact single-primitive basis)")
print("Config: 1s¹")
print("Grid: r0=1e-6, rmax=20 Bohr, N=2000")
print("=" * 56)
print(f"Electrons  : {N_elec:.10f}   (exact = 1)")
print(f"T_KS       : {T_KS:.10f}   (Koga  = 0.5000000000)")
print(f"T_vW       : {T_vW:.10f}   (= T_KS exactly for 1 orbital)")
print(f"Pauli KED  : min = {pauli.min():.2e}   (= 0 exactly: single orbital)")
print(f"             max = {pauli.max():.2e}")
print()
print("Per-orbital electron counts:")
ni = OCC_1S * R1s**2 / (4.0*pi)
ne = fmoment(0, 1.0/(4.0*pi), r, ni)
print(f"  1s (occ=1): {ne:.10f}   (exact = 1)")


# ============================================================================
# 10.  Save
# ============================================================================
np.savetxt(
    "H_Koga_radial_data.dat",
    np.column_stack((r, n, gradn, gradn2, lapl, tau_ks, tau_tf, s_sqr, q, tau_vw)),
    header=(
        "r   n   gradn   gradn2   lapl   tau_KS   tauTF   s2   q   tau_vW\n"
        f"H Z=1  Koga1999  E=-0.50000000 Ha  T_KS={T_KS:.10f} Ha\n"
        "Note: tau_KS = tau_vW exactly (single orbital) -> alpha_bar = 0 everywhere"
    )
)

np.savetxt(
    "H_Koga_orbitals.dat",
    np.column_stack((r, R1s, dR1s)),
    header=(
        "r  R1s  dR1s\n"
        "H Koga 1999: single 1s orbital (occ=1, l=0)\n"
        "R1s = 2*exp(-r) analytically (zeta=1, n=1 STO normalised)\n"
        "Column index: 0=r, 1=R1s, 2=dR1s"
    )
)
print()
print("Saved: H_Koga_radial_data.dat  (10 cols)")
print("Saved: H_Koga_orbitals.dat      (3 cols: r R1s dR1s)")


# ============================================================================
# 11.  Verify analytic form  R(r) = 2 exp(-r)
# ============================================================================
R_analytic = 2.0 * np.exp(-r)
max_err = np.max(np.abs(R1s - R_analytic))
print(f"\nMax deviation from analytic R(r)=2exp(-r): {max_err:.2e}  (should be ~0)")


# ============================================================================
# 12.  Plots
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("H (Z=1) — Koga 1999 Hartree-Fock STO (exact)", fontsize=14)

pairs = [
    (n,      r"$n(r)$",                True),
    (gradn2, r"$|\nabla n|^2$",        True),
    (lapl,   r"$\nabla^2 n$",          False),
    (tau_ks, r"$\tau_{KS} = \tau_W$",  True),
    (tau_tf, r"$\tau_{TF}$",           True),
    (s_sqr,  r"$s^2$ (reduced grad²)", True),
]

for ax, (y, lbl, logy) in zip(axes.flat, pairs):
    ax.plot(r, y, lw=1.8, color='steelblue')
    ax.set_xscale("log")
    if logy:
        pos = y[y > 0]
        if len(pos):
            ax.set_yscale("log")
    ax.set_xlabel("r  (Bohr)")
    ax.set_ylabel(lbl)
    ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("H_Koga_plots.png", dpi=150)
plt.show()
print("Saved: H_Koga_plots.png")
