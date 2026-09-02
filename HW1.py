import numpy as np
import matplotlib.pyplot as plt

# define cost function
f = lambda l, r: ((np.sqrt((2*l)/(1+l)) - 1) + 
                  (np.sqrt((2*r)/(l*(l+r))) - np.sqrt((2)/(l*(l+1)))) + 
                  (np.sqrt(1/r) - np.sqrt((2*l)/(r*(l+r)))))
g = lambda r: ((np.sqrt((2*r)/(1+r)) - 1) + 
               (np.sqrt(1/r) - np.sqrt((2)/(r*(1+r)))))
j = lambda l, r: f(l,r) - g(r)

l_iter = np.linspace(1,10,1000)
r = l_iter[-1]

costs = np.array([j(l, r) for l in l_iter])

# ---- plotting ------------------------------------------------------------
# palette (validated categorical/sequential set; slot 1 = blue)
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
 
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
 
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
 
# recessive gridlines behind everything
ax.grid(True, axis="both", color=GRIDLINE, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
 
# zero reference line -- the claim being tested is "curve never crosses this"
ax.axhline(0, color=BASELINE, linewidth=1.1, linestyle="--", zorder=1)
 
# shade the excess-cost region to emphasize strict positivity
ax.fill_between(l_iter, costs, 0, color=BLUE, alpha=0.12, zorder=1)
 
# the curve itself
ax.plot(l_iter, costs, color=BLUE, linewidth=2.4, zorder=3,
         label=r"$\Delta J(\ell,\,r)=J_{BE}-J_H$")
 
# mark the two degenerate endpoints where Delta J = 0 exactly
for l0 in (1, r):
    ax.plot(l0, 0, marker="o", markersize=7, markerfacecolor="#fcfcfb",
             markeredgecolor=BLUE, markeredgewidth=2, zorder=4)
 
# spines: keep only left/bottom, in muted ink
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(INK_MUTED)
    ax.spines[side].set_linewidth(1)
 
ax.tick_params(colors=INK_SECONDARY, labelsize=10.5)
 
ax.set_xlabel(r"$\ell$", fontsize=13, color=INK)
ax.set_ylabel(r"$\Delta J = J_{BE} - J_H$", fontsize=13, color=INK)
ax.set_title("Excess cost of the bielliptic transfer, $1 < \\ell < r$",
             fontsize=14, color=INK, pad=14, loc="left", fontweight="bold")
 
ax.text(0.02, 0.94, f"$r = {r:.0f}$", transform=ax.transAxes,
        fontsize=11, color=INK_SECONDARY, va="top")
 
ax.set_xlim(l_iter[0], l_iter[-1])
ax.legend(frameon=False, fontsize=10.5, loc="upper right",
          labelcolor=INK_SECONDARY)
 
fig.tight_layout()
plt.show()
