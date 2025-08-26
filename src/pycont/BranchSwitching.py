import numpy as np
import scipy.linalg as lg
import scipy.optimize as opt

def _find_all_zeros(f):
    t_range = np.linspace(0.0, 2.0*np.pi, 10**6 + 1)
    
    prev_f_val = f(np.array([1.0, 0.0]))
    roots = []
    for n in range(1, t_range.size):
        x = np.array([np.cos(t_range[n]), np.sin(t_range[n])])
        f_val = f(x)

        if f_val * prev_f_val <= 0.0:
            roots.append(x)
        prev_f_val = f_val

    return roots

# Minimizing the residual of a system is more stable than finding the exact nullspace
def _computeNullspace(Gu, Gp, M, r_diff):
    phi_0 = np.eye(M)[:,0]
    phi_objective = lambda y: 0.5*np.dot(Gu(y), Gu(y))
    phi_constraint = opt.NonlinearConstraint(lambda y: np.dot(y, y) - 1.0, 0.0, 0.0)
    min_result = opt.minimize(phi_objective, phi_0, constraints=(phi_constraint), method="BFGS", options={"eps": r_diff})
    phi = min_result.x

    w_objective = lambda y: np.sqrt(np.dot(Gu(y) + Gp, Gu(y) + Gp))
    min_result = opt.minimize(w_objective, np.zeros(M), method="BFGS", options={"eps": r_diff})
    w = min_result.x
    w_1 = np.append(w, 1.0)

    return phi, w, w_1

# Gu_v takes arguments u, p, v
def _computeCoefficients(Gu_v, Gp, x_s, phi, w, w_1, M, r_diff):

    # Compute a
    Gu_phi = lambda x: Gu_v(x[0:M], x[M], phi)
    a = np.dot(phi, (Gu_phi(x_s + r_diff * np.append(phi, 0.0)) - Gu_phi(x_s)) / r_diff)

    # Compute b
    Gx_w = lambda x: Gu_v(x[0:M], x[M], w) + Gp(x[0:M], x[M])
    b = np.dot(phi, (Gx_w(x_s + r_diff * np.append(phi, 0.0)) - Gx_w(x_s)) / r_diff)

    # Compute c
    c = np.dot(phi, (Gx_w(x_s + r_diff * w_1) - Gx_w(x_s)) / r_diff)

    return a, b, c

def _solveABSystem(a, b, c):
    f = lambda y: a*y[0]**2 + 2*b*y[0]*y[1] + c*y[1]**2
    solutions = _find_all_zeros(f)

    return solutions

def branchSwitching(G, Gu_v, Gp, x_s, x_prev, sp):
    print('\nBranch Switching')
    # Setting up variables
    M = x_s.size - 1
    u = x_s[0:M]
    p = x_s[M]

    # Computing necessary coefficients and vectors
    phi, w, w_1 =_computeNullspace(lambda v: Gu_v(u, p, v), Gp(u,p), M, sp["rdiff"])
    a, b, c = _computeCoefficients(Gu_v, Gp, x_s, phi, w, w_1, M, sp["rdiff"])
    solutions = _solveABSystem(a, b, c)

    # Fina all 4 branch tangents
    directions = []
    tangents = []
    for n in range(len(solutions)):
        alpha = solutions[n][0]
        beta  = solutions[n][1]

        s = 0.01
        N = lambda x: np.dot(alpha*phi + beta/np.sqrt(1.0)*w, x[0:M] - x_s[0:M]) + beta/np.sqrt(1.0)*(x[M] - x_s[M]) + s
        F_branch = lambda x: np.append(G(x[0:M], x[M]), N(x))

        tangent = np.append(alpha*phi + beta/np.sqrt(1.0)*w, beta/np.sqrt(1.0))
        x0 = x_s + s * tangent / lg.norm(tangent)
        dir = opt.newton_krylov(F_branch, x0, rdiff=sp["rdiff"], f_tol=sp["tolerance"])

        directions.append(dir)
        tangents.append(tangent)

    # Remove the direction where we came from
    inner_prodct = -np.inf
    for n in range(len(directions)):
        inner_pd = np.dot(directions[n]-x_s, x_prev-x_s) / (lg.norm(directions[n]-x_s) * lg.norm(x_prev-x_s))
        if inner_pd > inner_prodct:
            inner_prodct = inner_pd
            idx = n
    directions.pop(idx)
    tangents.pop(idx)
    print('Branch Switching Directions:', directions)

    # Returning 3 continuation directions
    return directions, tangents