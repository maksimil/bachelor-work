from matplotlib import pyplot as plt
import numpy as np
import scipy
import sys

pcr_data = []

filename = sys.argv[1]
outbasename = sys.argv[2]

with open(filename, "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        pcr_data.append(float(line[:-1]))

pcr_data = np.array(pcr_data)
maxcycles = pcr_data.shape[0] 

# Initial points

fl_mean = pcr_data.mean()
fl_mid = (np.max(pcr_data) - np.min(pcr_data))/2.0

mean_point = next(n for n in range(maxcycles) if pcr_data[n] >= fl_mean)
mid_point = next(n for n in range(maxcycles) if pcr_data[n] >= fl_mid)
suggested_p = [0.2, 0.4, 0.55, np.log(2)]

def OptLinear(p, n0):
    mat = np.zeros((maxcycles, 3))

    for n in range(maxcycles):
        mat[n,0] = 1
        mat[n,1] = n
        mat[n,2] = Sigmoid(p, n0, n)

    res = scipy.optimize.lsq_linear(mat, pcr_data)

    return res

def LogLinRegImprove(n0, p, withweight):
    f0, f1, a = OptLinear(p, n0).x

    mat = np.zeros((maxcycles, 2))
    b = np.zeros((maxcycles))

    for n in range(maxcycles):
        hatfn = pcr_data[n] - (f0+f1*n)
        val = a/hatfn-1.0

        if withweight:
            swn = np.abs(hatfn * (a-hatfn))/a
        else:
            swn = 1

        if val >= 1e-4:
            mat[n, 0] = swn
            mat[n, 1] = n * swn
            b[n] = np.log(val) * swn

    res = scipy.optimize.lsq_linear(mat, b)

    p = -res.x[1]
    n0 = res.x[0]/p

    return [n0, p]

mean_patent = np.array([[mean_point]*len(suggested_p), suggested_p]) 
mid_patent = np.array([[mid_point]*len(suggested_p), suggested_p]) 

# Optimization

def Sigmoid(p, n0, n):
    return 1.0/(1.0 + np.exp(-p*(n-n0)))

def SigmoidLoss(p, n0):
    res = OptLinear(p, n0)

    return np.sqrt(res.cost)/(pcr_data.max() - pcr_data.min())

sigmoid_res = scipy.optimize.minimize(
        lambda x: SigmoidLoss(x[0], x[1]), np.array([np.log(2), mid_point]),
        method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

opt_p, opt_n0 = sigmoid_res.x

opt_f0, opt_f1, opt_a = OptLinear(opt_p, opt_n0).x

print(f"Optimal params: p={opt_p}, n0={opt_n0}, f0={opt_f0}, f1={opt_f1}, a={opt_a}")
eff0 = Sigmoid(opt_p, opt_n0, 1)/Sigmoid(opt_p, opt_n0, 0)-1
effend = Sigmoid(opt_p, opt_n0, maxcycles-1)/Sigmoid(opt_p, opt_n0, maxcycles-2)-1
print(f"E={np.exp(opt_p)-1}, E_0={eff0}, drop={eff0 - effend}")

def OptSigmoid(n):
    return opt_f0 + opt_f1*n + opt_a*Sigmoid(opt_p, opt_n0, n)

cycles = np.arange(0, maxcycles, 1)
sigmoid_r2 = 1 - np.vectorize(lambda n: (OptSigmoid(n) - pcr_data[n])**2)(cycles).sum() / np.vectorize(lambda n: (pcr_data.mean() - pcr_data[n])**2)(cycles).sum()
rounded_r2 = np.floor(sigmoid_r2*1000)/1000

# Plotting

npoints = 100
tickstep = 10

figa, axa = plt.subplots(figsize=(8,1.6))
figb, axb = plt.subplots(figsize=(8,3.2))
figc, axc = plt.subplots(figsize=(4,3.2))
figd, axd = plt.subplots(figsize=(4,3.2))

sigmoid_data = np.vectorize(OptSigmoid)(cycles)
axa.plot(cycles, pcr_data, "ko-", mfc="none", label="$F_n$")
axa.plot(cycles, sigmoid_data, "rx--", label=f"$M_n^{{(S)}}$ $(R^2>{rounded_r2})$")

limcycles = max(maxcycles, opt_n0 + 5)
axa.legend(loc="upper left")
axa.set_xlim([-1, limcycles])

axa.set_xlabel("$n$")
axa.set_ylabel("Сигнал")

axa.grid()

def PlotCont(cycle0, cycle1, eff0, eff1, ax, lines):
    n0_arr = np.linspace(cycle0, cycle1, npoints)
    p_arr = np.log(np.linspace(eff0, eff1, npoints))

    n0_mesh, p_mesh = np.meshgrid(n0_arr, p_arr)

    z = np.vectorize(lambda n0, p: SigmoidLoss(p,n0))(n0_mesh, p_mesh)

    vmin = 0.1
    vmax = 1.1

    for key in lines.keys():
        cont = ax.contour(
                n0_mesh, p_mesh, z, cmap="magma", vmin=vmin, vmax=vmax,
                levels=lines[key], linestyles=key)
        ax.clabel(cont, inline=True, fontsize=10)

    ax.plot([opt_n0], [opt_p], "kD", label="G")

    ax.grid()

    ax.set_xlim([cycle0, cycle1])
    ax.set_ylim([np.log(eff0), np.log(eff1)])

    ax.set_xlabel("$n_0$")
    ax.set_ylabel("$p$")
    ax.legend()

P_MARKER = "gP"
PP_MARKER = "r^"

def DrawPoints(ax, weighted):
    mean_patent_improve = np.array(
            [LogLinRegImprove(n0, p, weighted) for [n0,p] in mean_patent.T]).T
    mid_patent_improve = np.array(
            [LogLinRegImprove(n0, p, weighted) for [n0,p] in mid_patent.T]).T

    # Patent points
    ax.plot(*mean_patent, P_MARKER, label="$P$")
    ax.plot(*mid_patent, PP_MARKER, label="$P'$")

    # Improve on patent points
    ax.plot(*mean_patent_improve, P_MARKER, mfc="none",
             label="$I_w$" if weighted else "$I$")
    ax.plot(*mid_patent_improve, PP_MARKER, mfc="none",
             label="$I_w'$" if weighted else "$I'$")

    # Draw lines
    for n in range(len(suggested_p)):
        ax.plot([mean_patent[0][n], mean_patent_improve[0][n]],
                [mean_patent[1][n], mean_patent_improve[1][n]], "k--", linewidth=1)
        ax.plot([mid_patent[0][n], mid_patent_improve[0][n]],
                [mid_patent[1][n], mid_patent_improve[1][n]], "k--", linewidth=1)

lines = { "solid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] }
#  , "dashed": [0.4, 0.5], "dashdot": [0.6, 0.7, 0.8], "dotted": [0.9, 1.0]}

axb.plot(*mean_patent, P_MARKER, label="$P$")
axb.plot(*mid_patent, PP_MARKER, label="$P'$")
DrawPoints(axc, False)
DrawPoints(axd, True)

local_lim0 = min(opt_n0-3, mid_point-1, mean_point-1)
local_lim1 = max(opt_n0+3, mid_point+1, mean_point+1)

PlotCont(-1, limcycles, 1.05, 3, axb, lines)
PlotCont(local_lim0, local_lim1, 1.05, 3, axc, lines)
PlotCont(local_lim0, local_lim1, 1.05, 3, axd, lines)

figa.subplots_adjust(left=0.08, right=0.98, bottom=0.3, top=0.95)
figb.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.95)
figc.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.95)
figd.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.95)

figa.savefig(f"{outbasename}_a.eps")
figb.savefig(f"{outbasename}_b.eps")
figc.savefig(f"{outbasename}_c.eps")
figd.savefig(f"{outbasename}_d.eps")
