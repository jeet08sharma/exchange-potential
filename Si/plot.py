import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator

# ---- Load data ----
r2scan     = np.loadtxt('../R2SCAN/r2scan.dat')
r2scan_LKT = np.loadtxt('../R2SCAN/r2scan_LKT.dat')
r2scan_PGS = np.loadtxt('../R2SCAN/r2scan_PGS.dat')
scan       = np.loadtxt('../SCAN/scan.dat')

r       = r2scan[:, 0]
vx      = r2scan[:, 1]
vx_lkt  = r2scan_LKT[:, 1]
vx_pgs  = r2scan_PGS[:, 1]
vx_scan = scan[:,1]

# ---- Global rc settings for paper-quality output ----
plt.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset" : "stix",
    "font.size"        : 12,
    "axes.labelsize"   : 14,
    "axes.titlesize"   : 14,
    "xtick.labelsize"  : 12,
    "ytick.labelsize"  : 12,
    "legend.fontsize"  : 11,
    "axes.linewidth"   : 1.2,
    "xtick.direction"  : "in",
    "ytick.direction"  : "in",
    "xtick.top"        : True,
    "ytick.right"      : True,
    "xtick.major.size" : 6,
    "ytick.major.size" : 6,
    "xtick.minor.size" : 3,
    "ytick.minor.size" : 3,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.minor.width": 0.9,
    "ytick.minor.width": 0.9,
    "legend.frameon"   : False,
})

# ---- Figure ----
fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=300)

ax.plot(r, r * vx,
        lw=2.2, color="#4B0082", ls="-",
        label=r"r$^2$SCAN")          # indigo

ax.plot(r, r * vx_lkt,
        lw=2.2, color="#0072B2", ls="-",
        label=r"r$^2$SCAN@LKT")      # scientific blue

ax.plot(r, r * vx_pgs,
        lw=2.2, color="#D55E00", ls="-",
        label=r"r$^2$SCAN@PGS")      # orange-red

ax.plot(r, r * vx_scan,
        lw=2.2, color="#009E73", ls="-",
        label="SCAN")                # teal-green

# Axes setup
ax.set_xscale("log")
ax.set_xlim(0.1, 10)
ax.set_ylim(-2.5,1)

ax.set_xlabel(r"$r$ (Bohr)")
ax.set_ylabel(r"$rv_x^{\mathrm{sl}}(r)$ (Ha)")

# Reference line at y = 0
ax.axhline(0.0, color="gray", lw=0.6, ls=":")

# Minor ticks
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

ax.legend(loc="lower left", handlelength=2.8, borderaxespad=0.6)

plt.tight_layout()
plt.savefig("Si_vx_r2scan_scan.pdf", bbox_inches="tight")
# plt.savefig("vx_r2scan.png", bbox_inches="tight", dpi=600)
plt.show()
