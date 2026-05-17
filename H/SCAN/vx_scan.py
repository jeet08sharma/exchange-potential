"""
vx_scan_xe_complete.py
======================
SCAN exchange potential for Xe — semilocal gKS decomposition.

Implements the functional derivative of the SCAN exchange functional
(Sun, Ruzsinszky, Perdew, PRL 2015):

    v_x^sl(r) = ∂(n·ε_x)/∂n  −  ∇·[∂(n·ε_x)/∂∇n]

References
----------
    Sun et al., PRL 115, 036402 (2015)
    Equations referenced as Eq.(N) below refer to that paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt


# 1.  Lagrange derivative engine


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
        xpp  = np.add.reduce(rmesh * Gpp)          
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


# 2.  Load data

data_d = np.loadtxt('..//H/H_Koga_radial_data.dat')
r      = data_d[:, 0]
n      = data_d[:, 1]
gradn  = data_d[:, 2]   
tau_ks = data_d[:, 5]  
tau_tf = data_d[:, 6]   
s2     = data_d[:, 7]   
tau_vw = data_d[:, 9]   

n_safe = np.maximum(n, 1e-16)
NN     = 5               

# 3.  SCAN constants  

Mu  = 10.0 / 81.0
b2  = sqrt(5913.0 / 405000.0)
b1  = (511.0 / 13500.0) / (2.0 * b2)   
b3  = 0.5
k1  = 0.065
b4  = Mu**2 / k1 - 1606.0 / 18225.0 - b1**2

hx0 = 1.174              
a1  = 4.9479             
c1x = 0.667
c2x = 0.8
dx  = 1.24
 
# 4.  LDA exchange  

CX_LDA = (3.0 / 4.0) * (3.0 / pi)**(1.0 / 3.0)
kF     = (3.0 * pi**2 * n_safe)**(1.0 / 3.0)
ex_lda = -CX_LDA * n_safe**(1.0 / 3.0)
dex_dn =  ex_lda / (3.0 * n_safe)

# 5.  Reduced gradient   

p      = np.maximum(s2, 1e-30)
s      = np.sqrt(p)
dp_dn  = -(8.0 / 3.0) * p / n_safe
dp_dgn =  2.0 * gradn / (4.0 * kF**2 * n_safe**2)
ds_dp  =  0.5 / np.maximum(s, 1e-30)


# 6.  SCAN iso-orbital indicator  

tau_pauli  = tau_ks - tau_vw
alpha      = tau_pauli / np.maximum(tau_tf, 1e-30)

dtau_vw_dn = -tau_vw / n_safe
dtau_tf_dn = (5.0 / 3.0) * tau_tf / n_safe

dalpha_dn   = ((-dtau_vw_dn) * tau_tf
               - tau_pauli   * dtau_tf_dn) / np.maximum(tau_tf**2, 1e-60)
dalpha_dgn  = -(gradn / (4.0 * n_safe)) / np.maximum(tau_tf, 1e-30)
dalpha_dtau =  1.0 / np.maximum(tau_tf, 1e-30)

# 7.  g_x(s) 

s_safe  = np.maximum(s, 1e-30)
exp_gx  = np.exp(-a1 / s_safe**0.5)
gxs     = 1.0 - exp_gx
dgxs_ds = np.nan_to_num(-(a1 / (2.0 * s_safe**1.5)) * exp_gx)
dgxs_dp = dgxs_ds * ds_dp

# 8.  x(s, α)  

exp_x1     = np.exp(-abs(b4) * p / Mu)
dexp_x1_dp = (-abs(b4) / Mu) * exp_x1

exp_x2 = np.exp(-b3 * (1.0 - alpha)**2)

dexp_x2_dn   = 2.0 * b3 * (1.0 - alpha) * exp_x2 * dalpha_dn
dexp_x2_dgn  = 2.0 * b3 * (1.0 - alpha) * exp_x2 * dalpha_dgn
dexp_x2_dtau = 2.0 * b3 * (1.0 - alpha) * exp_x2 * dalpha_dtau

Q   = b1 * p + b2 * (1.0 - alpha) * exp_x2
xsp = np.nan_to_num(Mu * p * (1.0 + (b4 * p / Mu) * exp_x1) + Q**2)

dxsp_dp = np.nan_to_num(
    Mu * (1.0 + (b4 * p / Mu) * exp_x1)           
  + Mu * p * ((b4 / Mu) * exp_x1                   
             + (b4 * p / Mu) * dexp_x1_dp)
  + 2.0 * Q * b1                                   
)

factor_alpha = -1.0 + 2.0 * b3 * (1.0 - alpha)**2

dQ_dn   = b2 * exp_x2 * dalpha_dn   * factor_alpha
dQ_dgn  = b2 * exp_x2 * dalpha_dgn  * factor_alpha
dQ_dtau = b2 * exp_x2 * dalpha_dtau * factor_alpha

dxsp_dn   = dxsp_dp * dp_dn   + 2.0 * Q * dQ_dn
dxsp_dgn  = dxsp_dp * dp_dgn  + 2.0 * Q * dQ_dgn
dxsp_dtau =                      2.0 * Q * dQ_dtau   

# 9.  h1_x(s, α)  

denom_h  = np.maximum((1.0 + xsp / k1)**2, 1e-60)
hx1      = np.nan_to_num(1.0 + k1 - k1 / (1.0 + xsp / k1))

dhx1_dp   = np.nan_to_num(dxsp_dp   / denom_h)
dhx1_dn   = np.nan_to_num(dxsp_dn   / denom_h)
dhx1_dgn  = np.nan_to_num(dxsp_dgn  / denom_h)
dhx1_dtau = np.nan_to_num(dxsp_dtau / denom_h)

# 10.  f_x(α)  

fx     = np.zeros_like(alpha)
dfx_da = np.zeros_like(alpha)

m_low = alpha <= 1.0
m_hig = alpha >  1.0

# Branch 1:
a_l      = alpha[m_low]
denom_l  = np.maximum(1.0 - a_l, 1e-30)            
t_l      = np.exp(-c1x * a_l / denom_l)
fx[m_low]     = t_l
dfx_da[m_low] = -c1x / denom_l**2 * t_l

# Branch 2:
a_h      = alpha[m_hig]
denom_h2 = np.minimum(1.0 - a_h, -1e-30)          
arg_h    = np.clip(c2x / denom_h2, -500.0, 0.0)    
t_h      = -dx * np.exp(arg_h)
fx[m_hig]     = t_h
dfx_da[m_hig] = t_h * c2x / denom_h2**2            

# 11.  Enhancement factor  F_x  [SCAN Eq. 8]


inner = hx1 + fx * (hx0 - hx1)        
Fx    = np.nan_to_num(inner * gxs)

dFx_dp = np.nan_to_num(
    (1.0 - fx) * dhx1_dp * gxs       
  + inner * dgxs_dp                   
)


dinner_dn   = np.nan_to_num(
    (1.0 - fx) * dhx1_dn   + dfx_da * dalpha_dn   * (hx0 - hx1))
dinner_dgn  = np.nan_to_num(
    (1.0 - fx) * dhx1_dgn  + dfx_da * dalpha_dgn  * (hx0 - hx1))
dinner_dtau = np.nan_to_num(
    (1.0 - fx) * dhx1_dtau + dfx_da * dalpha_dtau * (hx0 - hx1))

dFx_dn   = dinner_dn   * gxs    
dFx_dgn  = dinner_dgn  * gxs
dFx_dtau = dinner_dtau * gxs

# 12.  Semilocal exchange potential  v_x^sl 

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

# 14.  Diagnostics


print(f"α  range : {alpha.min():.4f} → {alpha.max():.4f}")
print(f"Fx range : {Fx.min():.4f}  → {Fx.max():.4f}  (should be ≤ 1.174)")
print(f"vx_sl    : {vx_sl.min():.4f} → {vx_sl.max():.4f}")
print(f"NaNs in vx_sl  : {np.isnan(vx_sl).sum()}")


# 15.  Plot


#fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#fig.suptitle('Xe (Z=54) — SCAN exchange potential', fontsize=13)

#ax = axes[0]
plt.plot(r,  vx_sl, color='steelblue', lw=2.0, label=r'$SCAN$')
plt.axhline(0, color='k', lw=0.8)
plt.xlim(0, 5);  plt.ylim(-10, 0.5)
#ax.set_xscale('log')
plt.xlabel(r'$r$ (Bohr)', fontsize=12)
plt.ylabel(r'$v_x^{sl}$ (Ha)', fontsize=12)
#plt.title('Semilocal part — Eq.(9) term', fontsize=10)
plt.legend(fontsize=10);  plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('H_vx_scan_complete.png', dpi=150)
plt.show()
print("Saved: H_vx_scan_complete.png")
