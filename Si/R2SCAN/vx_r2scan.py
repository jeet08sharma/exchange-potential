import numpy as np
import matplotlib.pyplot as plt
from math import pi,sqrt

#~~~~~~~~~~~~~~~~~~~~~~~~~1.Lagrange derivative~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _gridinterval(index, ir, nn):
    return max(0, ir - nn) - ir, min(index - 1, ir + nn) - ir

def _lightweights(s, rmesh, iorder = 1):
    Nb = len(rmesh)
    idx = np.arange(Nb, dtype=float)
    pisubi = np.zeros(Nb)
    for i in range(Nb):
        pii = float(i) - idx.copy()
        pii[i] = 1.0
        pisubi[i] = np.multiply.reduce(pii)

    pis = float(s) - idx; pis[s] = 1.0
    ipis = 1.0 / pis;   ipis[s] = 0.0
    st = np.add.reduce(ipis)
    Gp = (pisubi[s] / pisubi) * ipis; Gp[s] = st
    xp = np.add.reduce(rmesh * Gp)
    H = Gp/xp
    J = np.zeros(Nb)

    if iorder == 2:
        sq = np.add.reduce(ipis * ipis)
        Gpp = 2.0 * Gp * (st - ipis); Gpp[s] = st**2 - sq
        xpp = np.add.reduce(rmesh * Gp)
        J = (Gpp - H*xpp)/ xp**2
    return H,J

def localderivs(dens, rmesh, nn, iorder=1):
    N = len(dens)
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
    df, _ = localderivs(f, r, nn)
    return df + 2.0 * f / r

#
#~~~~~~~~~~~~~~~~~~~~~~~~~2.  Load data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data_d = np.loadtxt('../Si_Koga_radial_data.dat')
r      = data_d[:, 0]
n      = data_d[:, 1]
gradn  = data_d[:, 2]
gradn2 = data_d[:, 3]
tau_ks = data_d[:, 5]
tau_tf = data_d[:, 6]
s2     = data_d[:, 7]
tau_vw = data_d[:, 9]


n_safe = np.maximum(n, 1e-16)
NN = 5
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~3.r2SCAN constants~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CX = np.array([1.0, -0.667, -0.4445555, -0.663086601049,
               1.451297044490, -0.887998041597,
               0.234528941479, -0.023185843322])
C1X=0.667; C2X=0.8; DX=1.24; K0=0.174; K1=0.065
MU=10.0/81.0; A1=4.9479; DP2=0.361; ETA=0.001
H0X  = 1.0 + K0
CETA = 20.0/27.0 + ETA * 5.0/3.0
C2   = -sum(i * CX[i] * (1.0 - H0X) for i in range(1, 8))


#~~~~~~~~~~~~~~~~~~~~~~~~4.  LDA exchange~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CX_LDA = (3.0/4.0) * (3.0/pi)**(1.0/3.0)
kF     = (3.0 * pi**2 * n_safe)**(1.0/3.0)
ex_lda = -CX_LDA * n_safe**(1.0/3.0)
dex_dn =  ex_lda / (3.0 * n_safe)

#~~~~~~~~~~~~~~~~~~~~~5.  Reduced gradient p~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

p      = np.maximum(s2, 1e-30)
s = np.sqrt(p)
dp_dn  = -(8.0/3.0) * p / n_safe
dp_dgn = 2.0 * gradn / (4.0 * kF**2 * n_safe**2)

#~~~~~~~~~~~6.  ᾱ and derivatives for Eq. (9) — tau_KS treated as independent~~~~~~
tau_pauli = tau_ks - tau_vw
#denom_a   = tau_tf + ETA * tau_vw
#denom_a = tau_tf*np.exp(-40.0*p/27.0) + tau_vw  #PGS
lkt = -(1/np.cosh(1.3*s)) * np.tanh(1.3*s) * (1.3/(2*s))*dp_dn
denom_a = tau_tf*(1/np.cosh(1.3*s)) + tau_vw    #LKT


alpha_bar = tau_pauli / denom_a

dtau_vw_dn = -tau_vw / n_safe


#ddenom_dn  = ((5.0/3.0) * tau_tf - ETA * tau_vw) / n_safe
#ddenom_dn = (5.0/3.0)*(tau_tf/n_safe)*np.exp(-40.0*p/27.0) + tau_tf*np.exp(-40.0*p/27.0)*(-40.0*dp_dn/27) + dtau_vw_dn #PGS
ddenom_dn = (5.0/3.0)*(tau_tf/n_safe)*(1/np.cosh(1.3*s))+ tau_tf*lkt +dtau_vw_dn

dalpha_dn  = (-dtau_vw_dn * denom_a
              - tau_pauli * ddenom_dn) / np.maximum(denom_a**2, 1e-60)
dalpha_dgn = -(gradn / (4.0 * n_safe)) / np.maximum(denom_a, 1e-30)

dalpha_dtau = 1.0 / np.maximum(denom_a, 1e-30)

#~~~~~~~~~~~~~~~~~~~~~7.f_x(ᾱ) — three-branch~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fxalpha     = np.zeros_like(alpha_bar)
dfxalpha_da = np.zeros_like(alpha_bar)
m_neg = alpha_bar < 0.0
m_mid = (alpha_bar >= 0.0) & (alpha_bar <= 2.5)
m_pos = alpha_bar > 2.5

if np.any(m_neg):
    a = alpha_bar[m_neg]; t = np.exp(-C1X * a / (1.0 - a))
    fxalpha[m_neg] = t;  dfxalpha_da[m_neg] = -C1X / (1.0 - a)**2 * t
if np.any(m_mid):
    a = alpha_bar[m_mid]
    fxalpha[m_mid]     = sum(CX[i] * a**i for i in range(8))
    dfxalpha_da[m_mid] = sum(i * CX[i] * a**(i-1) if i > 0 else 0.0 for i in range(8))
if np.any(m_pos):
    a = alpha_bar[m_pos]; t = -DX * np.exp(C2X / (1.0 - a))
    fxalpha[m_pos] = t
    dfxalpha_da[m_pos] = t * C2X / (1.0 - a)**2

#~~~~~~~~~~~~~~~~~~~~~~~~~~8.g_x(p) and h1_x(p)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
exp_gx  = np.exp(-A1 / p**0.25)
gxp     = 1.0 - exp_gx
dgxp_dp = np.nan_to_num(-(A1 / (4.0 * p**1.25)) * exp_gx)

exp_h    = np.exp(-p**2 / DP2**4)
xp       = (CETA * C2 * exp_h + MU) * p
dxp_dp   = np.nan_to_num(CETA * C2 * exp_h * (1.0 - 2.0*p**2/DP2**4) + MU)
hx1p     = 1.0 + K1 - K1 / (1.0 + xp / K1)
dhx1p_dp = np.nan_to_num(dxp_dp / (1.0 + xp / K1)**2)

#~~~~~~~~~~~~~~~~~~9.Enhancement factor Fx and derivatives~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inner  = hx1p + fxalpha * (H0X - hx1p)
Fx     = inner * gxp
dFx_dp = np.nan_to_num(dhx1p_dp * (1.0 - fxalpha) * gxp + inner * dgxp_dp)
dFx_da = dfxalpha_da * (H0X - hx1p) * gxp

#~~~~~~~~~~~~~~~~~~10.v_xc^sl  (semilocal multiplicative potential)~~~~~~~~~~~~~~~~~~~~~

T1  = (ex_lda * Fx
       + n_safe * (dex_dn * Fx
                   + ex_lda * (dFx_dp * dp_dn  + dFx_da * dalpha_dn)))
A_r = n_safe * ex_lda * (dFx_dp * dp_dgn + dFx_da * dalpha_dgn)
vx_sl = T1 - radial_divergence(A_r, r, NN)
#~~~~~~~~~~~~~~~~~~~~~11. Saving the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.savetxt(
        "r2scan_LKT.dat",
        np.column_stack((r, vx_sl)),
        header=(
                "r  vx")
        )
print()
print("Saved:r2scan.dat")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~12.  Plot~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.plot(r,vx_sl,color='steelblue',lw=2.0,label="R2SCAN")
plt.xlim(0,10)
plt.ylim(-10,0.1)
plt.xlabel(r'$r$ (Bohr)', fontsize=12)
plt.ylabel(r'$v_x^{sl}$ (Ha)', fontsize=12)
#plt.xscale("log")
plt.grid(True)

plt.tight_layout()
plt.legend()
plt.savefig('Xe_vx_r2scan_sl.png', dpi=150)
plt.show()
print("Saved: Xe_vx_r2scan_sl.png")
