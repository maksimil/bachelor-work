import numpy as np
import scipy
import sys
import math

pcr_data = []

filename = "./data/Karlen_data_2.csv"

with open(filename, "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        pcr_data.append(float(line[:-1]))

pcr_data = np.array(pcr_data)
maxcycles = pcr_data.shape[0] 

def Model(f0, f1, a, p, n):
    return f0 + f1*n + a*np.exp(p*(n-(maxcycles-1)))

def OptLinear(p):
    mat = np.zeros((maxcycles, 3))

    for n in range(maxcycles):
        mat[n,0] = 1
        mat[n,1] = n
        mat[n,2] = np.exp(p*(n-(maxcycles-1)))

    res = scipy.optimize.lsq_linear(mat, pcr_data)

    return res

def ModelLoss(x):
    p = x[0]

    r = 0.0

    f0,f1,a = OptLinear(p).x

    for n in range(maxcycles):
        r += (pcr_data[n] - Model(f0,f1,a,p,n))**2

    return r

def CalcR2(prediction, data):
    data_mean = data.sum()/data.shape[0]

    rss = 0.0
    tss = 0.0

    for k in range(prediction.shape[0]):
        rss += (prediction[k] - data[k])**2
        tss += (data_mean - data[k])**2

    return 1.0 - rss/tss

res = scipy.optimize.minimize(
        ModelLoss, 0.5, 
        method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

opt_p = res.x[0]
opt_f0,opt_f1,opt_a = OptLinear(opt_p).x
print(f"{opt_p=}, {opt_f0=}, {opt_f1=}, {opt_a=}")

print(f"R2={CalcR2(np.vectorize(lambda n: Model(opt_f0, opt_f1, opt_a, opt_p, n))(np.arange(0, maxcycles, 1)), pcr_data)}")
