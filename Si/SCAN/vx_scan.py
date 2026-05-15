"""
vx_scan_xe_complete.py
======================
SCAN exchange potential for Xe — semilocal gKS decomposition.

Implements the functional derivative of the SCAN exchange functional
(Sun, Ruzsinszky, Perdew, PRL 2015):

    v_x^sl(r) = ∂(n·ε_x)/∂n  −  ∇·[∂(n·ε_x)/∂∇n]

    v_x^τ(r)  = n·ε_x^LDA · ∂F_x/∂τ        (orbital-specific gKS term)

References
----------
    Sun et al., PRL 115, 036402 (2015)
    Equations referenced as Eq.(N) below refer to that paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

# ============================================================================
# 1.  Lagrange derivative engine
# ============================================================================

def _gridinterval(index, ir, nn):
    return max(0, ir - nn) - ir, min(index - 1, ir + nn) - ir


def _lightweights(s, rmesh, iorder=1):
    Nb     = len(rmesh)
    idx    = np.arange(Nb, dtype=float)
    pisubi = np.zeros(Nb)
    for i in range(Nb):
        pii    = float(i) - idx.copy()
        pii[i] = 1.0
        pisubi[i] = np.multiply.reduce(pii)
    pis    = float(s) - idx;  pis[s]  = 1.0
    ipis   = 1.0 / pis;       ipis[s] = 0.0
    st     = np.add.reduce(ipis)
    Gp     = (pisubi[s] / pisubi) * ipis;  Gp[s] = st
    xp     = np.add.reduce(rmesh * Gp)
    H      = Gp / xp
    J      = np.zeros(Nb)
    if iorder == 2:
        sq   = np.add.reduce(ipis * ipis)
        Gpp  = 2.0 * Gp * (st - ipis);  Gpp[s] = st**2 - sq
        xpp  = np.add.reduce(rmesh * Gpp)          # uses Gpp, not Gp
        J    = (Gpp - H * xpp) / xp**2
    return H, J


def localderivs(dens, rmesh, nn, iorder=1):
    N    = len(dens)
    grad = np.zeros(N)
    lapl = np.zeros(N)
    for x in range(N):
        in1, in2 = _gridinterval(N, x, nn)
        nb = dens[x + in1: x + in2 + 1]
        rb = rmesh[x + in1: x + in2 + 1]
        H, J = _lightweights(-in1, rb, iorder=iorder)
        grad[x] = np.add.reduce(H * nb)
        if iorder == 2:
            lapl[x] = np.add.reduce(J * nb)
    return grad, lapl


def radial_divergence(f, r, nn):
    """∇·f in spherical radial coords: df/dr + 2f/r"""
    df, _ = localderivs(f, r, nn)
    return df + 2.0 * f / r


# ============================================================================
# 2.  Load data
# ============================================================================

data_d = np.loadtxt('/home/sanyam/Development/alphabeta/VXC/Final/Si/Si_Koga_radial_data.dat')
r      = data_d[:, 0]
n      = data_d[:, 1]
gradn  = data_d[:, 2]   # |∇n|
tau_ks = data_d[:, 5]   # KS kinetic energy density
tau_tf = data_d[:, 6]   # Thomas–Fermi KE density
s2     = data_d[:, 7]   # p = s² = |∇n|²/(4kF²n²)
tau_vw = data_d[:, 9]   # von Weizsäcker KE density

n_safe = np.maximum(n, 1e-16)
NN     = 5               # Lagrange stencil half-width

# ============================================================================
# 3.  SCAN constants  (Sun et al. 2015)
# ============================================================================

Mu  = 10.0 / 81.0
b2  = sqrt(5913.0 / 405000.0)
b1  = (511.0 / 13500.0) / (2.0 * b2)   # (511/13500) * 1/(2b2)
b3  = 0.5
k1  = 0.065
b4  = Mu**2 / k1 - 1606.0 / 18225.0 - b1**2

hx0 = 1.174              # h⁰_x  (single-orbital/α=0 limit)
a1  = 4.9479             # fitted to H atom exchange energy
c1x = 0.667
c2x = 0.8
dx  = 1.24

# ============================================================================
# 4.  LDA exchange
#     ε_x^LDA = −C_x · n^(1/3),   C_x = (3/4)(3/π)^(1/3)
#     ∂ε_x^LDA/∂n = ε_x^LDA / (3n)
# ============================================================================

CX_LDA = (3.0 / 4.0) * (3.0 / pi)**(1.0 / 3.0)
kF     = (3.0 * pi**2 * n_safe)**(1.0 / 3.0)
ex_lda = -CX_LDA * n_safe**(1.0 / 3.0)
dex_dn =  ex_lda / (3.0 * n_safe)

# ============================================================================
# 5.  Reduced gradient   p = s²,   s = |∇n|/(2kF n)
#
#     dp/dn  = −(8/3) p/n
#     dp/d|∇n| = 2|∇n| / (4kF²n²)
#     ds/dp  = 1/(2s)        [needed for g_x chain rule]
# ============================================================================

p      = np.maximum(s2, 1e-30)
s      = np.sqrt(p)
dp_dn  = -(8.0 / 3.0) * p / n_safe
dp_dgn =  2.0 * gradn / (4.0 * kF**2 * n_safe**2)
ds_dp  =  0.5 / np.maximum(s, 1e-30)

# ============================================================================
# 6.  SCAN iso-orbital indicator  α = τ_Pauli / τ_TF
#     (no η correction — that is r²SCAN)
#
#     ∂α/∂n    via quotient rule + ∂τ_vW/∂n = −τ_vW/n, ∂τ_TF/∂n = (5/3)τ_TF/n
#     ∂α/∂|∇n| via ∂τ_vW/∂|∇n| = |∇n|/(4n)
#     ∂α/∂τ_KS = 1/τ_TF
# ============================================================================

tau_pauli  = tau_ks - tau_vw
alpha      = tau_pauli / np.maximum(tau_tf, 1e-30)

dtau_vw_dn = -tau_vw / n_safe
dtau_tf_dn = (5.0 / 3.0) * tau_tf / n_safe

dalpha_dn   = ((-dtau_vw_dn) * tau_tf
               - tau_pauli   * dtau_tf_dn) / np.maximum(tau_tf**2, 1e-60)
dalpha_dgn  = -(gradn / (4.0 * n_safe)) / np.maximum(tau_tf, 1e-30)
dalpha_dtau =  1.0 / np.maximum(tau_tf, 1e-30)

# ============================================================================
# 7.  g_x(s) = 1 − exp(−a1/√s)   [SCAN Eq. 7]
#
#     NOTE: argument is s^(−1/2), NOT p^(−1/4) — those differ by a factor
#           because s = √p  →  s^(−1/2) = p^(−1/4), so numerically equal,
#           but derivatives must use ds/dp correctly.
#
#     dg/ds = −(a1 / (2s^(3/2))) * exp(−a1/√s)
#     dg/dp = dg/ds * ds/dp
# ============================================================================

s_safe  = np.maximum(s, 1e-30)
exp_gx  = np.exp(-a1 / s_safe**0.5)
gxs     = 1.0 - exp_gx
dgxs_ds = np.nan_to_num(-(a1 / (2.0 * s_safe**1.5)) * exp_gx)
dgxs_dp = dgxs_ds * ds_dp

# ============================================================================
# 8.  x(s, α)  [SCAN Eq. 6]
#
#     x = μ s² [1 + (b4 s²/μ) exp(−|b4|s²/μ)]
#           + [b1 s² + b2(1−α) exp(−b3(1−α)²)]²
#
#     using p = s²:
#     x = μ p  [1 + (b4 p/μ) exp_x1] + Q²
#     where:
#         exp_x1 = exp(−|b4|p/μ)
#         exp_x2 = exp(−b3(1−α)²)
#         Q      = b1 p + b2(1−α) exp_x2
#
#     Derivatives:
#       ∂x/∂p  — α independent of p
#       ∂x/∂{n,|∇n|,τ}  — only α changes (p-channel already in ∂x/∂p)
# ============================================================================

exp_x1     = np.exp(-abs(b4) * p / Mu)
dexp_x1_dp = (-abs(b4) / Mu) * exp_x1

exp_x2 = np.exp(-b3 * (1.0 - alpha)**2)

# ∂exp_x2/∂α = 2b3(1−α) exp_x2  →  chain to each density variable
dexp_x2_dn   = 2.0 * b3 * (1.0 - alpha) * exp_x2 * dalpha_dn
dexp_x2_dgn  = 2.0 * b3 * (1.0 - alpha) * exp_x2 * dalpha_dgn
dexp_x2_dtau = 2.0 * b3 * (1.0 - alpha) * exp_x2 * dalpha_dtau

Q   = b1 * p + b2 * (1.0 - alpha) * exp_x2
xsp = np.nan_to_num(Mu * p * (1.0 + (b4 * p / Mu) * exp_x1) + Q**2)

# ∂x/∂p  (α fixed — no spurious +1, only b1 from d[Q²]/dp)
dxsp_dp = np.nan_to_num(
    Mu * (1.0 + (b4 * p / Mu) * exp_x1)           # d/dp [μp · 1]
  + Mu * p * ((b4 / Mu) * exp_x1                   # d/dp [μp · (b4p/μ)exp_x1]
             + (b4 * p / Mu) * dexp_x1_dp)
  + 2.0 * Q * b1                                   # d/dp [Q²], only b1p varies
)

# ∂Q/∂{n,|∇n|,τ}  factored:
#   ∂/∂• [b2(1−α)exp_x2] = b2 exp_x2 ∂α/∂• [-1 + 2b3(1−α)²]
factor_alpha = -1.0 + 2.0 * b3 * (1.0 - alpha)**2

dQ_dn   = b2 * exp_x2 * dalpha_dn   * factor_alpha
dQ_dgn  = b2 * exp_x2 * dalpha_dgn  * factor_alpha
dQ_dtau = b2 * exp_x2 * dalpha_dtau * factor_alpha

dxsp_dn   = dxsp_dp * dp_dn   + 2.0 * Q * dQ_dn
dxsp_dgn  = dxsp_dp * dp_dgn  + 2.0 * Q * dQ_dgn
dxsp_dtau =                      2.0 * Q * dQ_dtau   # p independent of τ

# ============================================================================
# 9.  h¹_x(s, α)  [SCAN Eq. 5]
#
#     h¹ = 1 + k1 − k1/(1 + x/k1)
#     ∂h¹/∂x = 1/(1 + x/k1)²
# ============================================================================

denom_h  = np.maximum((1.0 + xsp / k1)**2, 1e-60)
hx1      = np.nan_to_num(1.0 + k1 - k1 / (1.0 + xsp / k1))

dhx1_dp   = np.nan_to_num(dxsp_dp   / denom_h)
dhx1_dn   = np.nan_to_num(dxsp_dn   / denom_h)
dhx1_dgn  = np.nan_to_num(dxsp_dgn  / denom_h)
dhx1_dtau = np.nan_to_num(dxsp_dtau / denom_h)

# ============================================================================
# 10.  f_x(α)  [SCAN Eq. 9 — two branches]
#
#      Branch 1 (α ≤ 1): f_x = exp[−c1x α/(1−α)]
#      Branch 2 (α > 1): f_x = −dx exp[c2x/(1−α)]
#                              note: 1−α < 0 here → argument ≤ 0 always
#
#      FIX: clamp denominators on the correct side to prevent overflow
# ============================================================================

fx     = np.zeros_like(alpha)
dfx_da = np.zeros_like(alpha)

m_low = alpha <= 1.0
m_hig = alpha >  1.0

# Branch 1: α ≤ 1  →  1−α > 0
a_l      = alpha[m_low]
denom_l  = np.maximum(1.0 - a_l, 1e-30)            # positive, safe
t_l      = np.exp(-c1x * a_l / denom_l)
fx[m_low]     = t_l
dfx_da[m_low] = -c1x / denom_l**2 * t_l

# Branch 2: α > 1  →  1−α < 0  →  c2x/(1−α) ≤ 0  → exp decays, never overflows
#           clamp denominator from the negative side
a_h      = alpha[m_hig]
denom_h2 = np.minimum(1.0 - a_h, -1e-30)           # negative, bounded away from 0
arg_h    = np.clip(c2x / denom_h2, -500.0, 0.0)    # argument always ≤ 0
t_h      = -dx * np.exp(arg_h)
fx[m_hig]     = t_h
dfx_da[m_hig] = t_h * c2x / denom_h2**2            # denom_h2² > 0

# ============================================================================
# 11.  Enhancement factor  F_x  [SCAN Eq. 8]
#
#      F_x = {h¹_x + f_x [h⁰_x − h¹_x]} · g_x(s)
#
#      Derivatives split into:
#        p-channel   : via ∂h¹/∂p and ∂g/∂p
#        α-channels  : via ∂h¹/∂{n,|∇n|,τ} and ∂f_x/∂α
#      g_x has no explicit α dependence.
# ============================================================================

inner = hx1 + fx * (hx0 - hx1)          # h¹ + fx(h0 − h1)
Fx    = np.nan_to_num(inner * gxs)

# ∂F_x/∂p  (s-channel, α fixed)
dFx_dp = np.nan_to_num(
    (1.0 - fx) * dhx1_dp * gxs          # ∂inner/∂p · g
  + inner * dgxs_dp                     # inner · ∂g/∂p
)

# ∂inner/∂{n,|∇n|,τ}  — α-channel only
#   ∂inner/∂• = (1−fx)·∂h¹/∂•  +  dfx/dα·∂α/∂•·(h0−h1)
dinner_dn   = np.nan_to_num(
    (1.0 - fx) * dhx1_dn   + dfx_da * dalpha_dn   * (hx0 - hx1))
dinner_dgn  = np.nan_to_num(
    (1.0 - fx) * dhx1_dgn  + dfx_da * dalpha_dgn  * (hx0 - hx1))
dinner_dtau = np.nan_to_num(
    (1.0 - fx) * dhx1_dtau + dfx_da * dalpha_dtau * (hx0 - hx1))

dFx_dn   = dinner_dn   * gxs      # g_x has no α dependence
dFx_dgn  = dinner_dgn  * gxs
dFx_dtau = dinner_dtau * gxs

# ============================================================================
# 12.  Semilocal exchange potential  v_x^sl  (same for all orbitals)
#
#      v_x^sl = ∂(n εx)/∂n  −  ∇·[∂(n εx)/∂∇n]
#
#      where εx = ε_x^LDA · F_x
#
#      ∂(n εx)/∂n = εx + n [∂ε_x^LDA/∂n · Fx
#                           + ε_x^LDA · (∂Fx/∂p · ∂p/∂n + ∂Fx/∂n)]
#                                         ↑ p-channel    ↑ α-channel
#
#      ∂(n εx)/∂|∇n| = n · ε_x^LDA · (∂Fx/∂p · ∂p/∂|∇n| + ∂Fx/∂|∇n|)
# ============================================================================

ex   = ex_lda * Fx

T1   = np.nan_to_num(
    ex_lda * Fx
  + n_safe * (dex_dn * Fx
             + ex_lda * (dFx_dp * dp_dn + dFx_dn))
)

A_r  = np.nan_to_num(
    n_safe * ex_lda * (dFx_dp * dp_dgn + dFx_dgn)
)

vx_sl = T1 - radial_divergence(A_r, r, NN)

np.savetxt(
        "scan.dat",
        np.column_stack((r, vx_sl)),
        header=(
                "r  vx")
        )
print()
print("Saved:scan.dat")

# ============================================================================
# 14.  Diagnostics
# ============================================================================

print(f"α  range : {alpha.min():.4f} → {alpha.max():.4f}")
print(f"Fx range : {Fx.min():.4f}  → {Fx.max():.4f}  (should be ≤ 1.174)")
print(f"vx_sl    : {vx_sl.min():.4f} → {vx_sl.max():.4f}")
print(f"NaNs in vx_sl  : {np.isnan(vx_sl).sum()}")

# ============================================================================
# 15.  Plot
# ============================================================================

#fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#fig.suptitle('Xe (Z=54) — SCAN exchange potential', fontsize=13)

#ax = axes[0]
plt.plot(r,  vx_sl, color='steelblue', lw=2.0, label=r'$SCAN$')
plt.axhline(0, color='k', lw=0.8)
plt.xlim(0.1, 5);  plt.ylim(-10, 0.5)
#ax.set_xscale('log')
plt.xlabel(r'$r$ (Bohr)', fontsize=12)
plt.ylabel(r'$v_x^{sl}$ (Ha)', fontsize=12)
#plt.title('Semilocal part — Eq.(9) term', fontsize=10)
plt.legend(fontsize=10);  plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Xe_vx_scan_complete.png', dpi=150)
plt.show()
print("Saved: Xe_vx_scan_complete.png")
