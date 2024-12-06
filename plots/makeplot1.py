from matplotlib import pyplot as plt
import numpy as np
import math
import scipy

pcr_data = []

with open("./data/Karlen_data.csv", "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        pcr_data.append(float(line[:-1]))

pcr_data = np.array(pcr_data)
maxcycles = pcr_data.shape[0] 

# Optimization

def Sigmoid(x, n):
    f0,f1,a,p,n0 = x
    return f0 + f1*n + a/(1.0 + math.exp(-p*(n-n0)))

def SigmoidLoss(x):
    r = 0

    for n in range(maxcycles):
        r += (pcr_data[n]-Sigmoid(x,n))**2

    return r

def SigmoidEff(x, n):
    f0,f1,a,p,n0 = x

    xp = np.array([0,0,a,p,n0])

    return Sigmoid(xp, n+1)/Sigmoid(xp, n) - 1.0

sigmoid_res = scipy.optimize.minimize(
        SigmoidLoss, np.array([0,0,pcr_data[-1],0.69,27]), 
        method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

sigmoid_params = sigmoid_res.x

def Algebraic(x, n):
    f0,f1,a,b,n0 = x

    back = f0 + f1*n
    shape = 0.0

    if (n > 0):
        shape = a / (1.0 + (n/n0)**(-(b+1.0)))

    return back + shape

def AlgebraicLoss(x):
    r = 0

    for n in range(maxcycles):
        r += (pcr_data[n]-Algebraic(x,n))**2

    return r

def AlgebraicEff(x, n):
    if (n == 0):
        return np.inf

    f0,f1,a,b,n0 = x
    xp = [0,0,a,b,n0]

    return Algebraic(xp,n+1)/Algebraic(xp,n)-1.0


algebraic_res = scipy.optimize.minimize(
        AlgebraicLoss, np.array([0,0,pcr_data[-1],1.5,27]), 
        method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

algebraic_params = algebraic_res.x

print(f"{sigmoid_params=}, r={sigmoid_res.fun}")
print(f"{algebraic_params=}, r={algebraic_res.fun}")

def CalcR2(prediction, data):
    data_mean = data.sum()/data.shape[0]

    rss = 0.0
    tss = 0.0

    for k in range(prediction.shape[0]):
        rss += (prediction[k] - data[k])**2
        tss += (data_mean - data[k])**2

    return 1.0 - rss/tss

# Plotting

# Plot A

cycles = np.arange(0, maxcycles, 1)

figa, axa = plt.subplots(figsize=(8,8/3))
figb, axb = plt.subplots(figsize=(8,8*2/3))

sigmoid_data = np.vectorize(lambda n: Sigmoid(sigmoid_params, n))(cycles)
algebraic_data = np.vectorize(lambda n: Algebraic(algebraic_params, n))(cycles)

print(f"SR2 = {CalcR2(sigmoid_data, pcr_data)}")
print(f"AR2 = {CalcR2(algebraic_data, pcr_data)}")

axa.plot(cycles, pcr_data, "ko-", mfc="none", label="$F_n$")
axa.plot(cycles, algebraic_data, "k^-.", label="$M^{(A)}_n$")
axa.plot(cycles, sigmoid_data, "rx--", label="$M^{(S)}_n$")

axa.grid()
axa.legend()
axa.set_xlabel("$n$")
axa.set_ylabel("Сигнал")

axa.set_xlim([-0.5, maxcycles-0.5])

# Plot B

f0 = sigmoid_params[0]
f1 = sigmoid_params[1]

hat_pcr_eff = np.vectorize(lambda n: (pcr_data[n+1] - (f0 + f1*n)) / (pcr_data[n] - (f0 + f1*n)) - 1)(cycles[:-1])
sigmoid_eff = np.vectorize(lambda n: SigmoidEff(sigmoid_params, n))(cycles)
algebraic_eff = np.vectorize(lambda n: AlgebraicEff(algebraic_params, n))(cycles)

print(f"SER2(>=15) = {CalcR2(hat_pcr_eff[15:], sigmoid_eff[15:])}")
print(f"AER2(>=15) = {CalcR2(hat_pcr_eff[15:], algebraic_eff[15:])}")

print(f"SER2(>=20) = {CalcR2(hat_pcr_eff[20:], sigmoid_eff[20:])}")
print(f"AER2(>=20) = {CalcR2(hat_pcr_eff[20:], algebraic_eff[20:])}")

axb.plot(cycles[:-1], hat_pcr_eff, "ko-", mfc="none", label="$\hat F_{n+1}/\hat F_n-1$")
axb.plot(cycles, algebraic_eff, "k^-.", label="$E^{(A)}_n$")
axb.plot(cycles, sigmoid_eff, "rx--", label="$E^{(S)}_n$")

axb.set_xlim([-0.5, maxcycles-0.5])
axb.set_ylim([-0.25,1.25])

axb.grid()
axb.legend()
axb.set_xlabel("$n$")
axb.set_ylabel("Эффективность")

figa.subplots_adjust(left=0.11, right=0.98, bottom=0.2, top=0.95)
figb.subplots_adjust(left=0.11, right=0.98, bottom=0.1, top=0.975)

figa.savefig("./eps/plot1_a.eps")
figb.savefig("./eps/plot1_b.eps")
