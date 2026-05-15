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
vx_scan = scan[:, 1]

# ---- Clip rSCAN to suppress divergence spikes ----
rvx_scan_clipped = np.clip(r * vx_scan, -1.75, 0.5)

# ---- Global rc settings for paper-quality output ----
plt.rcParams.update({
    "font.family"       : "serif",
    "font.serif"        : ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset"  : "stix",
    "font.size"         : 12,
    "axes.labelsize"    : 14,
    "axes.titlesize"    : 14,
    "xtick.labelsize"   : 12,
    "ytick.labelsize"   : 12,
    "legend.fontsize"   : 11,
    "axes.linewidth"    : 1.2,
    "xtick.direction"   : "in",
    "ytick.direction"   : "in",
    "xtick.top"         : True,
    "ytick.right"       : True,
    "xtick.major.size"  : 6,
    "ytick.major.size"  : 6,
    "xtick.minor.size"  : 3,
    "ytick.minor.size"  : 3,
    "xtick.major.width" : 1.1,
    "ytick.major.width" : 1.1,
    "xtick.minor.width" : 0.9,
    "ytick.minor.width" : 0.9,
    "legend.frameon"    : False,
})

# ---- Figure ----
fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=300)

# Each curve: unique color + unique dash pattern + unique linewidth combination
# so they remain distinguishable even in grayscale print
ax.plot(r,  vx,lw=2.2, color="#1f77b4",ls="--",label=r"r$^2$SCAN",zorder=4)

ax.plot(r,  vx_lkt,lw=2.0, color="#d62728",ls=(0, (8, 2)),label=r"r$^2$SCAN@LKT",zorder=3)

ax.plot(r,  vx_pgs,lw=2.0, color="#2ca02c",ls=(0, (4, 1, 1, 1)),label=r"r$^2$SCAN@PGS",zorder=2)

ax.plot(r, vx_scan,
        lw=1.5, color="#9467bd",
        #ls=(0, (2, 2)),           # short evenly-spaced dots
        alpha=0.85,
        label=r"SCAN",
        zorder=1)

# ---- Axes ----
#ax.set_xscale("log")
ax.set_xlim(0.01, 5)
ax.set_ylim(-10.0, 0.6)

ax.set_xlabel(r"$r$ (Bohr)")
ax.set_ylabel(r"$v_x^{\mathrm{sl}}(r)$ (Ha)")

ax.axhline(0.0, color="gray", lw=0.6, ls=":")

# Minor ticks
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

ax.legend(loc="lower right", handlelength=1.5, borderaxespad=0.4)

plt.tight_layout()
plt.savefig("Xenolog_vx_scan.pdf", bbox_inches="tight")
#plt.savefig("vx_r2scan_scan_fixed.png", bbox_inches="tight", dpi=600)
plt.show()

