import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def prob2():
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

def prob4():
    delta_i = np.pi / 2
    f = lambda r: (2*((np.sqrt((2*r) / (1 + r))) - 1) +
                   2*(np.sqrt((2) / (r*(1 + r))))*np.sin(delta_i/2) - 2*np.sin(delta_i/2))

    r_iter = np.linspace(1,10,1000)
    costs = np.array([f(r) for r in r_iter])

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(7, 5))
    plt.plot(r_iter, costs, lw=2)
    plt.axhline(0, color='k', linewidth=0.8)
    plt.xlabel(r'$r = r_2/r_1$')
    plt.ylabel(r'excess cost  $\bar c_{\mathrm{bielliptic}} - \bar c_{\mathrm{direct}}$')
    plt.title(r'Restricted bielliptic plane change vs. direct plane change, $\Delta i = 90^\circ$')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    g = lambda r, di: np.sqrt((np.sqrt((2*r) / (1 + r)) - 1)**2 + 
                              (4*(np.sqrt((2*r) / (1 + r)))*np.sin(di/2)**2))

    h = lambda r, di: 2*(np.sqrt((2) / (r*(1 + r))))*np.sin(di/2)

    j = lambda r, di1, di2, di3: g(r, di1) + h(r, di2) + g(r, di3)

    di_total = np.pi/2
    r_list = [2, 5, 7, 9, 10, 100]
    N = 500
    j_total = lambda r, di1, di3: j(r, di1, di_total - di1 - di3, di3)

    def plot_di(DI1, DI3, Z, r):
            fig, ax = plt.subplots(figsize=(7, 6))
            cf = ax.contourf(DI1, DI3, Z, levels=25, cmap='viridis')
            fig.colorbar(cf, ax=ax, label=r'total cost $J_{\mathrm{total}}$')
            cs = ax.contour(DI1, DI3, Z, levels=8, colors='white', linewidths=0.6)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
            ax.plot([0, 45], [0, 45], 'r--', lw=1.5, label=r'$\Delta i_1=\Delta i_3$')
            ax.set_xlabel(r'$\Delta i_1$ (deg)')
            ax.set_ylabel(r'$\Delta i_3$ (deg)')
            ax.set_title(rf'Dogleg cost contour, $r={r}$, $\Delta i_{{\mathrm{{total}}}}=90^\circ$')
            ax.set_xlim(0, 90); ax.set_ylim(0, 90)
            ax.set_aspect('equal')
            ax.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

    for r in r_list:
        di1_deg = np.linspace(0, 90, N)
        di3_deg = np.linspace(0, 90, N)
        DI1, DI3 = np.meshgrid(di1_deg, di3_deg)

        Z = j_total(r, np.radians(DI1), np.radians(DI3))
        Z = np.where(DI1 + DI3 <= 90, Z, np.nan)
        plot_di(DI1, DI3, Z, r)
        i, k = np.unravel_index(np.nanargmin(Z), Z.shape)
        print(f"min J_total @ r={r} = {Z[i,k]:.5f} at di1 = {DI1[i,k]:.2f} deg, di3 = {DI3[i,k]:.2f} deg")

def prob5_test():

    def T_of_l(l):
        return np.sqrt(2.0 * l / (1.0 + l))

    def J1_of(eta, l, di):
        T = T_of_l(l)
        return np.sqrt((T - 1.0) ** 2 + 4.0 * T * np.sin(eta * di / 2.0) ** 2)

    def J2_of(eta, l, di):
        T = T_of_l(l)
        return 2.0 * (T / l) * np.sin((1.0 - eta) * di / 2.0)

    def J_of(eta, l, di):
        return J1_of(eta, l, di) + J2_of(eta, l, di)

    def dJdeta(eta, l, di):
        T = T_of_l(l)
        J1 = J1_of(eta, l, di)
        return T * di * np.sin(eta * di) / J1 - (T / l) * di * np.cos((1.0 - eta) * di / 2.0)

    def eta_plot_test(l, di):

        eta_array = np.linspace(0,1,1000)
        F_array = np.array([dJdeta(eta, l, di) for eta in eta_array])
        J_array = np.array([J_of(eta, l, di) for eta in eta_array])

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"J or F = $\partial J / \partial \eta$")
        ax.set_title(rf"Cost J and $\partial J / \partial \eta$ vs. "
                    rf"$\eta$ Parameter for l = {l}, $\Delta i$ = {np.rad2deg(di):.0f} deg")
        ax.plot(eta_array, F_array, color='blue', label = r"$\partial J / \partial \eta$")
        ax.plot(eta_array, J_array, color='green', label = "Cost J")
        ax.axhline(y=0, color='red', linestyle='dotted')
        ax.legend()

        try:
            print(f"first root: {brentq(dJdeta, 0.0, 0.5, args=(l, di))}")
        except:
            print(f"no first root")
        try:
            print(f"second root: {brentq(dJdeta, 0.5, 1.0, args=(l, di))}")
        except:
            print(f"no second root")
        return fig

    fig1 = eta_plot_test(1.4, np.pi/2)
    fig2 = eta_plot_test(5, np.pi/2)
    fig3 = eta_plot_test(1.4468, np.deg2rad(5))

    fig1.show()
    fig2.show()
    fig3.show()

def prob5():

    def T_of_l(l):
        return np.sqrt(2.0 * l / (1.0 + l))

    def J1_of(eta, l, di):
        T = T_of_l(l)
        return np.sqrt((T - 1.0) ** 2 + 4.0 * T * np.sin(eta * di / 2.0) ** 2)

    def J2_of(eta, l, di):
        T = T_of_l(l)
        return 2.0 * (T / l) * np.sin((1.0 - eta) * di / 2.0)

    def J_of(eta, l, di):
        return J1_of(eta, l, di) + J2_of(eta, l, di)

    def dJdeta(eta, l, di):
        T = T_of_l(l)
        J1 = J1_of(eta, l, di)
        return T * di * np.sin(eta * di) / J1 - (T / l) * di * np.cos((1.0 - eta) * di / 2.0)

    def eta_star(l, di):

        f0 = dJdeta(0.0, l, di)
        f1 = dJdeta(1.0, l, di)
        # check if one endpoint is negative
        if f0 * f1 < 0.0:
            return brentq(dJdeta, 0.0, 1.0, args=(l, di))

        # otherwise do a grid search and pick the root going negative -> positive
        grid = np.linspace(0.0, 1.0, 10)
        vals = dJdeta(grid, l, di)
        sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
        # check for the case with no root on the interval
        if len(sign_changes) == 0:
            return 1.0 if f0 < 0.0 else 0.0
        
        min_idx = sign_changes[vals[sign_changes] < 0]
        a, b = grid[min_idx], grid[min_idx + 1]
        return brentq(dJdeta, a, b, args=(l, di))

    L_MIN = 1.05    # there's a singularity at l = 1, eta = 0
    L_MAX = 100.0
    N_L = 200
    DI_DEG = np.linspace(5.0, 90.0, 18)

    l_vals = np.logspace(np.log10(L_MIN), np.log10(L_MAX), N_L)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("viridis")

    for k, di_deg in enumerate(DI_DEG):
        di = np.deg2rad(di_deg)
        etas = np.empty_like(l_vals)
        for i, l in enumerate(l_vals):
            etas[i] = eta_star(l, di)
        color = cmap(k / (len(DI_DEG) - 1))
        ax.plot(l_vals, etas, lw=2, color=color, label=rf"$\Delta i = {di_deg:.0f}^\circ$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$l = r_2 / r_1$")
    ax.set_ylabel(r"$\eta^*$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Optimal plane-change split vs. orbit size ratio")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title=r"$\Delta i$", fontsize=9, title_fontsize=9, ncol=2)
    fig.tight_layout()
    fig.show()

def prob7():

    def J_burn1(l):
        return np.abs(np.sqrt((2*l)/(l+1)) - 1)

    def J_esc(l, x):
        return np.sqrt(2/l + x **2)

    def J_burn2(l, x):
        return J_esc(l, x) - np.sqrt(2/(l*(1+l)))

    def J(l, x):
        return J_burn1(l) + J_burn2(l, x)

    def plot_J_contour(L, X, Z):
        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ax.contourf(L, X, Z, levels=100, cmap='viridis')
        fig.colorbar(cf, ax=ax, label=r'total cost $J_{\mathrm{total}}$')
        cs = ax.contour(L, X, Z, levels=25, colors='white', linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
        ax.vlines(1, ymin=0, ymax=2, color='r', linestyle='--', lw=1.5, label=r'$l = 1$')
        ax.hlines(np.sqrt(2), xmin=0, xmax=1, color='k', linestyle='--', 
                  lw=1.5, label=r'$x = \sqrt{2}$')
        ax.set_xscale('log')
        ax.set_xlabel(r'$l = \frac{r_2}{r_1}$')
        ax.set_ylabel(r'$x = V_\infty / \sqrt{\frac{\mu}{r}}$')
        ax.set_title(rf'Escape Maneuver Optimality')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_J_circ(x):
        l = 1
        Jvec = J(l, x)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(x, Jvec, lw=2, color='k', label=rf"$Circular Cost J$")
        ax.vlines(np.sqrt(2), ymin=Jvec.min(), ymax=Jvec.max(), color='k', linestyle='--', 
                          lw=1.5, label=r'$x = \sqrt{2}$')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel(r'$x = V_\infty / \sqrt{\frac{\mu}{r}}$')
        ax.set_ylabel(r"Cost J @ l = 1")
        ax.set_title(r"Cost for One-Impulse Escape")
        plt.tight_layout()
        plt.show()

    def plot_J_curves(l, x_range, x, l_range):

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        ax1, ax2 = axes
        cmap = plt.get_cmap("viridis")

        for k, x_curr in enumerate(x_range):
            Jvec = J(l, x_curr)
            color = cmap(k / (len(x_range) - 1))
            ax1.plot(l, Jvec, lw=2, color=color, label=rf"$x = {x_curr:.2f}$")

        J_set = J(l, np.sqrt(2))
        ax1.plot(l, J_set, lw=2, color='r', linestyle='--', label=r"$x = \sqrt{2}$", alpha=0.5)
        
        ax1.set_xscale("log")
        ax1.set_xlabel(r'$l = \frac{r_2}{r_1}$')
        ax1.set_ylabel(r"Cost J")
        ax1.set_title(r"Cost vs. $l=\frac{r_2}{r_1}$ For Various x")
        ax1.grid(True, which="both", alpha=0.25)
        ax1.legend(title=r"$x$", fontsize=9, title_fontsize=9, ncol=2)

        for k, l_curr in enumerate(l_range):
            Jvec = J(l_curr, x)
            color = cmap(k / (len(l_range) - 1))
            ax2.plot(x, Jvec, lw=2, color=color, label=rf"$l = {l_curr:.2f}$")

        J_set = J(1, x)
        ax2.plot(x, J_set, lw=2, color='r', linestyle='--', label=r"$l = 1$", alpha=0.5)
        ax2.vlines(np.sqrt(2), ymin=J_set.min(), ymax=1.1, color='k', linestyle='--', 
                   lw=1.5, label=r'$x = \sqrt{2}$')
        
        ax2.set_xlabel(r'$x = V_\infty / \sqrt{\frac{\mu}{r}}$')
        ax2.set_ylabel(r"Cost J")
        ax2.set_title(r"Cost vs. $x = V_\infty / \sqrt{\frac{\mu}{r}}$ For Various l")
        ax2.grid(True, which="both", alpha=0.25)
        ax2.legend(title=r"$l$", fontsize=9, title_fontsize=9, ncol=2)


        plt.tight_layout()
        plt.show()

    def find_argmins(l, x):
        J_star = np.zeros(len(x))
        l_star = np.zeros(len(x))
        for k, x_curr in enumerate(x):
            Jvec = J(l, x_curr)
            min_idx = np.argmin(Jvec)
            J_star[k] = Jvec[min_idx]
            l_star[k] = l[min_idx]
        return (J_star, l_star)

    def plot_argmin(l, x):
        J_star, l_star = find_argmins(l, x)
        J_c = J(1,x)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        ax1, ax2 = axes
        ax1.plot(x, J_star, lw=2, color='b', label=r'$J^*(x) = min_l J(l,x)$')
        ax1.plot(x, J_c, lw=2, color='r', linestyle='--', label=r'$J(1,x)$', alpha=0.5)
        ax1.hlines(1, xmin=x.min(), xmax=x.max(), color='k', linestyle='--', 
                          lw=1.5, label=r'$J = 1$', alpha=0.5)
        ax1.set_xlabel('x')
        ax1.set_ylabel(r'$J^*(x) = min_l J(l,x)$')
        ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax1.legend()

        ax2.plot(x, l_star, lw=2, color='b', label=r'$l^*(x) = argmin_l J(l,x)$')
        ax1.hlines(1, xmin=x.min(), xmax=x.max(), color='k', linestyle='--', 
                                  lw=1.5, label=r'$l^*= 1$', alpha=0.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel(r'$l^*(x) = argmin_l J(l,x)$')
        ax2.grid(True, color='gray', linestyle='--', linewidth=0.5)

        fig.suptitle("Escape Maneuver Optimality: Cost and Optimal Transfer Geometry")
        plt.tight_layout()
        plt.show()



    N = 1000
    max_L = 100
    max_x = 2
    l = np.logspace(-2, np.log10(max_L), N)
    x = np.linspace(0, max_x, N)
    L, X = np.meshgrid(l, x)

    Z = J(L, X)
    #plot_J_contour(L, X, Z)

    #plot_J_circ(x)

    x_range = np.linspace(0, max_x, 21)
    l_range = np.logspace(-2, np.log10(max_L), 20)
    plot_J_curves(l, x_range, x, l_range)

    plot_argmin(l,x)

    

if __name__ == "__main__":
    # prob2()
    # prob4()
    # prob5_test()
    # prob5()
    prob7()