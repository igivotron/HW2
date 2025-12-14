import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


plotMach = False
plotTS = True

# Flight conditions
T0 = 216
R = 287
gamma = 1.4
piC = 20
Tt4 = 1400
Tt7 = 2300
etaC = 0.9
etaT = 0.9
c0 = sqrt(gamma * R * T0)

cp = gamma * R / (gamma - 1)

print("cp =", cp)

tau_lbdAB = Tt7 / T0
tau_lbd = Tt4 / T0

tau_c = piC**((gamma - 1) / (gamma * etaC))

Mtrans = 2.44


def getTau_r(M):
    return 1 + (gamma - 1) / 2 * M**2

def getTau_t(M):
    tau_r = getTau_r(M)
    return 1 - (tau_r/tau_lbd) * (tau_c -1)

def ST_turbojet(M):
    tau_t = getTau_t(M)
    tau_r = getTau_r(M)

    term1 = 2 / (gamma - 1)
    term2 = 1 - (1 / ( tau_r * (tau_c**etaC) * (tau_t**(1/etaT)) ))
    term3 = term1 * term2 * tau_lbdAB
    if term3 < 0:
        return None
    return c0 * (sqrt(term3) - M)

def ST_ramjet(M):
    tau_r = getTau_r(M)
    term1 = 2 / (gamma - 1)
    term2 = 1 - (1 / ( tau_r ))
    term3 = term1 * term2 * tau_lbdAB
    return c0 * (sqrt(term3) - M)

if plotMach:
    M_values = np.linspace(2, 4, 100)
    ST_turbojet_values = [ST_turbojet(M) for M in M_values]
    ST_ramjet_values = [ST_ramjet(M) for M in M_values]
    plt.figure(figsize=(10, 6))
    plt.plot(M_values, ST_turbojet_values, label='Turbojet', color='blue')
    plt.plot(M_values, ST_ramjet_values, label='Ramjet', color='red')
    plt.axvline(x=Mtrans, color='green', linestyle='--', label='Transition Mach Number (M=2.44)')
    plt.xlabel('Mach Number (M)', fontsize=14)
    plt.ylabel('Specific Thrust (ST) [m/s]', fontsize=14)
    plt.title('Specific Thrust vs Mach Number for Turbojet and Ramjet', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('Specific_Thrust_vs_Mach_Number.pdf')
    plt.show()


if plotTS:
    Npoints = 100
    M = 0.8
    tau_r = getTau_r(M)
    tau_t = getTau_t(M)
    tau_c = piC**((gamma - 1) / (gamma * etaC))
    pi_r = tau_r**(gamma / (gamma - 1))
    pi_t = tau_t**(gamma*etaT / (gamma - 1))


    # 0 to 2 Ram effect: isentropic compression
    s02 = np.zeros(Npoints)
    p02 = np.linspace(1, pi_r, Npoints)
    T02 = T0 * p02**((gamma - 1) / gamma)
    # 2 to 3 Compressor: polytropic compression
    p23 = np.linspace(pi_r, pi_r * piC, Npoints)
    T23 = T02[-1] * (p23 / p02[-1])**((gamma - 1) / (gamma * etaC))
    s23 = s02[-1] + cp * np.log(T23 / T02[-1]) - R * np.log(p23 / p02[-1])
    # 3 to 4 Combustion: isobaric heat addition
    p34 = np.full(Npoints, p23[-1])
    T34 = np.linspace(T23[-1], Tt4, Npoints)
    s34 = s23[-1] + cp * np.log(T34 / T23[-1])
    # 4 to 5 Turbine: polytropic expansion
    p45 = np.linspace(p34[-1], p34[-1] * pi_t, Npoints)
    T45 = T34[-1] * (p45 / p34[-1])**((gamma - 1) * etaT / gamma)
    s45 = s34[-1] + cp * np.log(T45 / T34[-1]) - R * np.log(p45 / p34[-1])
    # 5 to 7 : Afterburner: isobaric heat addition
    p57 = np.full(Npoints, p45[-1])
    T57 = np.linspace(T45[-1], Tt7, Npoints)
    s57 = s45[-1] + cp * np.log(T57 / T45[-1])

    M = 2.44
    tau_r = getTau_r(M)
    tau_t = getTau_t(M)
    tau_c = piC**((gamma - 1) / (gamma * etaC))
    pi_r = tau_r**(gamma / (gamma - 1))
    pi_t = tau_t**(gamma*etaT / (gamma - 1))

    s02prim = np.zeros(Npoints)
    p02prim = np.linspace(1, pi_r, Npoints)
    T02prim = T0 * p02prim**((gamma - 1) / gamma)
    # 2 to 3 Compressor: polytropic compression
    p23prim = np.linspace(pi_r, pi_r * piC, Npoints)
    T23prim = T02prim[-1] * (p23prim / p02prim[-1])**((gamma - 1) / (gamma * etaC))
    s23prim = s02prim[-1] + cp * np.log(T23prim / T02prim[-1]) - R * np.log(p23prim / p02prim[-1])
    # 3 to 4 Combustion: isobaric heat addition
    p34prim = np.full(Npoints, p23prim[-1])
    T34prim = np.linspace(T23prim[-1], Tt4, Npoints)
    s34prim = s23prim[-1] + cp * np.log(T34prim / T23prim[-1])
    # 4 to 5 Turbine: polytropic expansion
    p45prim = np.linspace(p34prim[-1], p34prim[-1] * pi_t, Npoints)
    T45prim = T34prim[-1] * (p45prim / p34prim[-1])**((gamma - 1) * etaT / gamma)
    s45prim = s34prim[-1] + cp * np.log(T45prim / T34prim[-1]) - R * np.log(p45prim / p34prim[-1])
    # 5 to 7 : Afterburner: isobaric heat addition
    p57prim = np.full(Npoints, p45prim[-1])
    T57prim = np.linspace(T45prim[-1], Tt7, Npoints)
    s57prim = s45prim[-1] + cp * np.log(T57prim / T45prim[-1])

    plt.figure(figsize=(10, 6))
    
    plt.plot(s02, T02, label='0-2 Ram Effect (M=0.8)', color='blue')
    plt.plot(s23, T23, label='2-3 Compressor (M=0.8)', color='orange')
    plt.plot(s34, T34, label='3-4 Combustion (M=0.8)', color='red')
    plt.plot(s45, T45, label='4-5 Turbine (M=0.8)', color='green')
    plt.plot(s57, T57, label='5-7 Afterburner (M=0.8)', color='purple')

    plt.plot(s02prim, T02prim, '--', label='0-2 Ram Effect (M=2.44)', color='blue')
    plt.plot(s23prim, T23prim, '--', label='2-3 Compressor (M=2.44)', color='orange')
    plt.plot(s34prim, T34prim, '--', label='3-4 Combustion (M=2.44)', color='red')
    plt.plot(s45prim, T45prim, '--', label='4-5 Turbine (M=2.44)', color='green')
    plt.plot(s57prim, T57prim, '--', label='5-7 Afterburner (M=2.44)', color='purple')


    plt.xlabel('Entropy (s) [J/(kg·K)]', fontsize=14)
    plt.ylabel('Temperature (T) [K]', fontsize=14)
    plt.title('T-s Diagram of Turbojet with Afterburner', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('TSM.pdf')
    plt.show()

    