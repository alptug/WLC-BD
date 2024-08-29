"""
WLC-BD Simulation
Author: Alptuğ Ulugöl
Date: 29/08/2024
This script performs a simulation of the Worm-Like Chain (WLC) model for a magnetic bead system. It calculates the force-extension relation and simulates traces for different forces using the Milstein solver. The simulation parameters and global constants are defined at the beginning of the script.
The main functions in this script are:
- wlcfunc(ext, Lp, Lc, T): Calculates the force-extension relation for the WLC model.
- faxenzcorrection_perp(z, r): Calculates the Faxén correction for a sphere of radius r at a distance z from a wall (perpendicular direction).
- faxenzcorrection_par(z, r): Calculates the Faxén correction for a sphere of radius r at a distance z from a wall (parallel direction).
- wlcext(f): Calculates the extension for a given force using the WLC model.
- kappa(f): Calculates the kappa parameter for a given force using the WLC model.
- simulate_trace(fmag, fg, gamma, dt, N): Simulates a trace for a given force using the Milstein solver.
- simtrc(fmag): Wrapper function for simulating a trace with a specific force.
The simulation results are saved in a pickle file named 'tracesimulations.pickle'.
Note: This script requires the autograd, scipy, and multiprocessing libraries to be installed.
"""


import autograd.numpy as np
from autograd import value_and_grad, grad
from scipy.optimize import root_scalar
from multiprocessing import Pool


########################################
# Global Parameters for the simulation #
########################################

# Number of cores to use
Ncores = 4

# Number of simulations with distinct forces
Nsim = 20

# Number of repetitions for each force
Nrepeat = 1

# Simulation time in seconds
timesim = 30.#s

# Minimum and maximum force in pN
fmagmin = 1.#pN
fmagmax = 80.#pN

########################################
#     Parameters for the WLC model     #
########################################

# Temperature in Kelvin
T = 293. # K
kT = 1.3806503e-2*T

# Contour length in nm
Lc = 7000 # nm

# Persistence length in nm
Lp = 45 # nm

########################################
#   Parameters for the Magnetic bead   #
########################################
# Drag coefficient in pN s/nm
gamma = 3e-5

# Bead radius in nm
rad = 1400.

iofrac = 0.26 # ironoxide fraction
den_io = 5.2e-12 # density of ironoxide in ng/nm^3
den_ps = 1.05e-12 # density of polystyrene in ng/nm^3
den_w  = 1e-12 # density of water in ng/nm^3

effective_bead_den = iofrac * den_io + (1. - iofrac) * den_ps - den_w

g = 9.81 # gravitational acceleration in m/s^2
#Fg = (4./3) * np.pi * rad ** 3 * effective_bead_den * g # effective gravitational force in pN
Fg = 0.0 # we set the gravitational force to zero which renders Fmag as the total force, since Fg is always subtracted from Fmag

########################################
#       End of Global Parameters       #
########################################


# 7-paraeter WLC model coefficients
wlccoef = np.array([-0.5164228, -2.737418, 16.07497, -38.87607, 39.49944, -14.17718])

# 7-parameter WLC model force-extension relation
def wlcfunc(ext, Lp, Lc, T):
    kT = 1.3806503e-2*T # k_B T in units pn nm
    z_scaled = ext/Lc
    a = np.array([1,-0.5164228,-2.737418,16.07497,-38.87607,39.49944,-14.17718])
    Fwlc = 1./(4.*(1. - z_scaled)**2)  - 0.25
    for i in range(7):
        Fwlc += a[i] * z_scaled**(i+1)
    return Fwlc * kT/Lp


# Faxén correction for a sphere of radius r at a distance z from a wall
fzcoef = np.array([1, -9/8, 0.5, -57/100, 1/5, 7/200, -1/25])
fzpow = np.array([0,1,3,4,5,11,12])
def faxenzcorrection_perp(z,r):
    x = r/z
    return 1./(np.power(x,fzpow)@fzcoef)

fpcoef = np.array([1, -9/16, 1/8])
fppow = np.array([0,1,3])
def faxenzcorrection_par(z,r):
    x = r/z
    return 1./(np.power(x,fppow)@fpcoef)


# Convenience functions for the WLC model
fwlc = lambda x: wlcfunc(x, Lp, Lc, T)
dfwlc = grad(fwlc)
d2fwlc = grad(fwlc,2)

fwlcvec = lambda r: - wlcfunc(np.sqrt(r @ r), Lp, Lc, T) * r / np.sqrt(r @ r)
#dfwlcvec = grad(fwlcv)

ffaxperp = lambda x: faxenzcorrection_perp(x, rad)
ffaxpar = lambda x: faxenzcorrection_par(x, rad)

def wlcext(f):
    return root_scalar(lambda x: (fwlc(x) - f), bracket=[0.,Lc-1e-8], fprime = dfwlc, fprime2=d2fwlc).root

def kappa(f):
    return dfwlc(wlcext(f))

forces = np.linspace(fmagmin, fmagmax,Nsim)
forces = np.repeat(forces,Nrepeat)
maxext = wlcext(fmagmax-Fg)
fcmax  = kappa(fmagmax-Fg) / (gamma*2*np.pi* ffaxperp(maxext+rad)) 
dt = 0.01/fcmax
N = int(timesim/dt)
time = np.arange(N)*dt

def milsteinSolver(alpha, betaperp,betapar, dt, N, x0):

    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size = (N,3))
    beta_dbeta_par = value_and_grad(betapar)
    beta_dbeta_perp = value_and_grad(betaperp)

    res = np.empty([N,3])
    res[0] = x0

    for i in range(N-1):
        xi = res[i]
        bpar, dbpar = beta_dbeta_par(xi[2])
        bperp, dbperp = beta_dbeta_perp(xi[2])
        b = np.array([bpar,bpar,bperp])
        db = np.array([dbpar,dbpar,dbperp])
        dWi = dW[i]
        res[i+1] = xi + alpha(xi) * dt + b * dWi + 0.5 * bperp * db * (dWi*dWi[2] - dt)

    return res

def simulate_trace(fmag,fg, gamma, dt ,N):
    assert(fmag > fg)
    x0 = np.array([0,0,wlcext(fmag-fg)])
    def f_det(x):
        f = fwlcvec(x)
        f[2] += fmag - fg
        f /= gamma
        projz = x[2]/np.sqrt(x@x)
        f[0] /= ffaxpar(x[2] + rad * projz)
        f[1] /= ffaxpar(x[2] + rad * projz)
        f[2] /= ffaxperp(x[2] + rad * projz)

        return f

    def f_sto_par(x):
        return np.sqrt(2. * kT  / (gamma * ffaxpar(x)))

    def f_sto_perp(x):
        return np.sqrt(2. * kT  / (gamma * ffaxperp(x)))

    return milsteinSolver(f_det, f_sto_perp,f_sto_par, dt, N, x0)

def simtrc(fmag):
    print(f'Simulating {fmag} pN.')
    res = simulate_trace(fmag,Fg, gamma, dt ,N)
    print(f'Done {fmag} pN.')
    return res





import pickle as pk
if __name__ == '__main__':
    print("Highest corner frequency in the simulation set: ",fcmax," Hz")
    print("Time step is set to ",dt,"s using the highest corner frequency.")
    print("Number of data points per trace: ",N)
    
    res = {'N':N, 'dt':dt, 'time':time, 'forces':forces, 'sims':[]}
    print(forces)
    with Pool(Ncores) as p:
        res['sims'] = p.map(simtrc, forces)

    with open('tracesimulations.pickle', 'wb') as f:
            pk.dump(res,f,pk.HIGHEST_PROTOCOL)
