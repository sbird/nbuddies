from ICs import generate_plummer_initial_conditions
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.style as mplstyle
mplstyle.use('fast')
from matplotlib import rc #
rc("font", family="DejaVu Sans", weight="normal", size="9") # bryance used Liberation Serif, swapping to DejaVu Sans cuz lib serif not found
rc("axes", grid=True)
rc("grid", linestyle="--")
rc("xtick", direction="in")
rc("ytick", direction="in")
rc("savefig", format="png", bbox="tight")

nbuddies_path = os.path.dirname(os.path.realpath(__file__))
GG = 4.301e-6 # Newton constant km^2 kpc / Msun s^2

def test_plummer_ICs():
    if not os.path.exists(nbuddies_path+"/diagnostics"):
        os.makedirs(nbuddies_path+"/diagnostics")
    
    vert = 1
    horz = 2
    gs = gridspec.GridSpec(vert, horz)
    fig = plt.figure(figsize=(horz*3, vert*3), dpi=300)
    
    N = int(1e6)
    a = 10*np.random.rand()
    m = 1e6
    blackholes = generate_plummer_initial_conditions(N, m, a)[0]

    r_magnitudes = np.array([np.linalg.norm(bh.position) for bh in blackholes])

    ax = fig.add_subplot(gs[0, 0])

    hist, bins, rects = ax.hist(r_magnitudes, bins=50, range=[0,2*a], label="IC Code")

    def shell_vol(r_1, r_2):
        return 4*np.pi*(r_2**3 - r_1**3)/3

    density = np.zeros(len(hist))
    r_points = np.array([(bins[i] + bins[i+1])/2 for i in range(len(hist))])
    for n in range(len(hist)):
        r = rects[n]
        density[n] = (r.get_height()/shell_vol(r.get_x(), r.get_x() + r.get_width())) * m
        r.set_height(density[n])

    def plummer_rho(r):
        return (3*m*N*a**2)/(4*np.pi*(a**2 + r**2)**(5/2))

    ax.plot(r_points, plummer_rho(r_points), label="theoretical")
    ax.vlines(x=a, ymin=0, ymax=1.1*plummer_rho(0), colors='r', linestyles='--', label="scale radius")

    ax.set_title("density")
    ax.set_xlabel("r (kpc)")
    ax.set_ylabel(r"$\rho \, \left(\frac{M_\odot}{kpc^3}\right)$")
    ax.set_ylim([0, 1.1*plummer_rho(0)])
    ax.legend()

    ax = fig.add_subplot(gs[0,1])

    v_disp = np.zeros(len(hist))
    v_points = np.array([(bins[i] + bins[i+1])/2 for i in range(len(hist))])
    for i in range(len(hist)):
        mask = (r_magnitudes >= bins[i]) & (r_magnitudes < bins[i+1])
        velocities = np.array([bh.velocity for bh in blackholes])[mask]
        v_disp[i] = np.sum(np.var(velocities, axis=0, ddof=0)) / 3

    ax.plot(v_points, v_disp, label="IC Code")
    def plummer_vdisp(r):
        return GG*m*N/(6*np.sqrt(a**2 + r**2))
    ax.plot(v_points, plummer_vdisp(v_points), label="theoretical")
    ax.vlines(x=a, ymin=0, ymax=1.1*plummer_vdisp(0), colors='r', linestyles='--', label="scale radius")
    
    ax.set_title("Velocity Dispersion")
    ax.set_xlabel("r (kpc)")
    ax.set_ylabel(r"$\sigma^2 \, \left(\frac{km}{s}\right)$")
    ax.set_ylim([0, 1.1*plummer_vdisp(0)])
    ax.legend()

    plt.tight_layout()
    plt.savefig(nbuddies_path+"/diagnostics/plummer_ICs_test.png")

    #I prune the first entry as it's prone to large flucuations since the bin is small
    density_errors = density - plummer_rho(r_points)
    density_rms_error = np.sqrt(np.sum(density_errors[1:]**2) / len(density_errors[1:]))
    
    v_disp_errors = v_disp - plummer_vdisp(v_points)
    v_disp_rms_error = np.sqrt(np.sum(v_disp_errors[1:]**2) / len(v_disp_errors[1:]))

    print(density_rms_error/plummer_rho(0), v_disp_rms_error/plummer_vdisp(0))

    assert density_rms_error/plummer_rho(0) < 0.01, "IC density doesn't match expectation"
    assert v_disp_rms_error/plummer_vdisp(0) < 0.01, "IC velocity dispersion doesn't match expectation"