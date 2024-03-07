#!/usr/bin/env python
# coding: utf-8


import numpy as np
import sys
import os
import matplotlib.pyplot as plt

data = np.genfromtxt(os.path.join(sys.path[0], "Vandermonde.txt"),
                     comments='#', dtype=np.float64)

# Define the Vandermonde matrix
x = data[:, 0]
y = data[:, 1]

xx = np.linspace(x[0], x[-1], 1001) # x values to interpolate at
V = np.array([[xi**i for i in range(len(x))] for xi in x])


def LU_decomp(A):
    """Perform LU decomposition on matrix A."""
    N = len(A)
    L, U = np.zeros((N, N)), np.zeros((N, N))

    for i in range(N):
        # Set diagonal equal to 1
        L[i, i] = 1

    for i in range(N):
        # Upper matrix
        for j in range(i, N):
            U[0, i] = A[0, i]  # set condition
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        # Lower matrix
        for j in range(i + 1, N):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
            
    return L, U


def solve_LU(L, U, y):
    """Solve for c in Vc = y."""
    n = len(y)
    # Solve Ly = b for y (forward substitution)
    b = np.zeros(n)
    for i in range(n):
        b[i] = y[i] - np.dot(L[i, :i], b[:i])
        
    # Solve Uc = y for c (backward substitution)
    c = np.zeros(n)
    for i in range(n-1, -1, -1):
        c[i] = (b[i] - np.dot(U[i, i+1:], c[i+1:])) / U[i, i]
        
    return c


def polynomial(x, c):
    """Evaluate the polynomial at the specified x points given coefficients c."""
    y = 0
    for i in range(len(c)):
        y += c[i] * x ** i
        
    return y


# Perform LU decomposition on V
L, U = LU_decomp(V)

# Calculate coeffiecients c
c = solve_LU(L, U, y)
# Save to .txt file
with open('c_coeffs.txt', 'a') as f:
    for i in range(len(c)):
        print(f'c_{i} = {c[i]}', file=f)

# Generate points for plotting
yya = polynomial(xx, c)
ya = polynomial(x, c)

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
axs = gs.subplots(sharex=True, sharey=False)
axs[0].plot(x, y, marker='o', linewidth=0)
plt.xlim(-1, 101)
axs[0].set_ylim(-400, 400)
axs[0].set_ylabel('$y$')
axs[1].set_ylim(1e-16, 1e1)
axs[1].set_ylabel('$|y-y_i|$')
axs[1].set_xlabel('$x$')
axs[1].set_yscale('log')
line, = axs[0].plot(xx, yya, color='orange')
line.set_label('Via LU decomposition')
axs[0].legend(frameon=False, loc="lower left")
axs[1].plot(x, abs(y-ya), color='orange')
plt.savefig('./plots/my_vandermonde_sol_2a.png', dpi=600)

