import numpy as np
import pandas as pd
import copy
np.set_printoptions(suppress=True, precision=6)  # Round my answers to 6 decimal places

# HW1
def get_lr(df):
    """
    get log return
    """
    log_return = np.log((df.pct_change().dropna() + 1).values)
    return log_return

def get_Sig(T):
    """
    get the sample cov Sigma given matrix T
    """
    T_bar = T - T.mean(axis=0)
    Sig = 1 / (T_bar.shape[0] - 1) * T_bar.T @ T_bar
    return Sig

def get_pctchg(df):
    """
    get percentage return
    """
    pct_chg = df.pct_change().dropna().values
    return pct_chg

def cov_log(df):
    """
    covariance matrix of log returns
    """
    T = get_lr(df)
    cov = get_Sig(T)
    return cov

def corr_log(df):
    """
    correlation matrix of log returns
    """
    Sig = cov_log(df)
    D_sig = np.diag(np.sqrt(np.diag(Sig)))
    D_sig_inv = np.linalg.inv(D_sig)
    corr = D_sig_inv @ Sig @ D_sig_inv
    return corr

def cov_pct(df):
    """
    covariance matrix of percentage returns
    """
    T = get_pctchg(df)
    cov = get_Sig(T)
    return cov

def corr_pct(df):
    """
    correlation matrix of percentage returns
    """
    Sig = cov_pct(df)
    D_sig = np.diag(np.sqrt(np.diag(Sig)))
    D_sig_inv = np.linalg.inv(D_sig)
    corr = D_sig_inv @ Sig @ D_sig_inv
    return corr

# HW2
def forward_subset(L, b):
    """
    Forward Substitution
    Input:
    - L: nonsingular lower triangular matrix of size n
    - b: column vector of size n
    Output:
    - x: solution to Lx=b
    """
    n = len(b)
    x = [0 for _ in range(n)]
    x[0] = b[0] / L[0][0]
    for j in range(1, n):
        ssum = 0
        for k in range(0, j):
            ssum += L[j][k] * x[k]
        x[j] = (b[j] - ssum) / L[j][j]
    
    for i in range(n):
        x[i] = round(x[i], 6)

    return x

def backward_subset(U, b):
    """
    Backward substitution
    Input:
    - U: nonsingular upper triangular matrix of size n
    - b: column vector of size n
    Output:
    - x: solution to Ux=b
    """
    n = len(b)
    x = [0 for _ in range(n)]
    x[n-1] = b[n-1] / U[n-1][n-1]
    for j in range(n-2, -1, -1):
        ssum = 0
        for k in range(j+1, n):
            ssum += U[j][k] * x[k]
        x[j] = (b[j] - ssum) / U[j][j]
    
    for i in range(n):
        x[i] = round(x[i], 6)
        
    return x

def lu_no_pivoting(A):
    """
    LU decomposition without pivoting
    Input:
    - A: nonsingular matrix of size n
    Output:
    - L, U
    """
    n = len(A)
    # create empty L, U
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(0, n-1):
        for l in range(i, n):
            U[i][l] = A[i][l]
            L[l][i] = A[l][i] / U[i][i]
        for j in range(i+1, n):
            for k in range(i+1, n):
                A[j][k] -= L[j][i] * U[i][k]
    L[n-1][n-1] = 1
    U[n-1][n-1] = A[n-1][n-1]

    for j in range(n):
        for k in range(n):
            L[j][k] = round(L[j][k], 6)
            U[j][k] = round(U[j][k], 6)

    return L, U

def linear_solve_LU_no_pivoting(A, b):
    """
    Input:
    - A: nonsingular square matrix of size n with LU decom
    - b: column vector of size n
    Output:
    - x: solution to Ax=b
    """
    L, U = lu_no_pivoting(A)
    y = forward_subset(L, b)  # solve Ly=b
    x = backward_subset(U, y)  # solve Ux=y
    return x

# HW3
def lu_row_pivoting(A): #@save
    """
    LU decomposition with row pivoting
    input: A
    output: P, L, U
    """
    n = A.shape[0]
    # initialize
    P, L = np.eye(n), np.eye(n)
    U = np.eye(n)
    for i in range(0, n-1):
        i_max = np.argmax(np.abs(A[i:n, i])) + i
        # switch rows i and i_max of A
        vv = copy.deepcopy(A[i, i:n])
        A[i, i:n] = A[i_max, i:n]
        A[i_max, i:n] = vv
        # update matrix P
        cc = copy.deepcopy(P[i])
        P[i] = P[i_max]
        P[i_max] = cc
        if i>0:
            ww = copy.deepcopy(L[i, 0:(i-1)])
            L[i, 0:(i-1)] = L[i_max, 0:(i-1)]
            L[i_max, 0:(i-1)] = ww
        for j in range(i, n):
            L[j, i] = A[j, i] / A[i, i]
            U[i, j] = A[i, j]
        for j in range(i+1, n):
            for k in range(i+1, n):
                A[j, k] -= L[j,i] * U[i, k]
    L[n-1, n-1] = 1
    U[n-1, n-1] = A[n-1, n-1]
    return P, L, U