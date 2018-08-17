#!/usr/bin/python
# -*- coding: utf-8 -*-
# code for PAIF manuscript
# FvW 05/2018

import os, sys #, pickle
import numpy as np
import matplotlib.pyplot as plt

# --- define logarithm
#log = np.log
log = np.log2


def H_1(x, ns):
    """Shannon entropy of the symbolic sequence x with ns symbols.

    Args:
        x: symbolic sequence, symbols = [0, 1, ..., ns-1]
        ns: number of symbols
    Returns:
        h: Shannon entropy of x
    """

    n = len(x)
    p = np.zeros(ns) # symbol distribution
    for t in range(n):
        p[x[t]] += 1.0
    p /= n
    h = -np.sum(p[p>0]*log(p[p>0]))
    return h


def H_2(x, y, ns):
    """Joint Shannon entropy of the symbolic sequences X, Y with ns symbols.

    Args:
        x, y: symbolic sequences, symbols = [0, 1, ..., ns-1]
        ns: number of symbols
    Returns:
        h: Shannon entropy of x
    """

    if (len(x) != len(y)):
        print("H_2 warning : sequences of different lengths, using the shorter...")
    n = min([len(x), len(y)])
    p = np.zeros((ns, ns)) # joint distribution
    for t in range(n):
        p[x[t],y[t]] += 1.0
    p /= n
    h = -np.sum(p[p>0]*log(p[p>0]))
    return h


def H_k(x, ns, k):
    """Shannon's joint entropy from x[t:t+k]
    x: symbolic sequence, symbols = [0, 1, ..., ns-1]
    ns: number of symbols
    k: length of k-history
    """

    N = len(x)
    f = np.zeros(tuple(k*[ns]))
    for t in range(N-k): f[tuple(x[t:t+k])] += 1.0
    f /= (N-k) # normalize distribution
    h_k = -np.sum(f[f>0]*log(f[f>0]))
    #m = np.sum(f>0)
    #h_k = h_k + (m-1)/(2*N) # Miller-Madow bias correction
    return h_k


def ais(x, ns, k=1):
    """Active information storage (AIS):

    *** Symbols must be 0, 1, 2, ... to use as indices ***

    \sum p(x[t+1], x_k[t]) log p(x[t+1], x_k[t]) / (p(x[t+1]) p(x_k[t]))
    \sum f_ij log (f_ij)/ (f_i f_j)
    i: x[t+1]  j: x[t], x[t-1], ..., x[t-k+1]

    I(X_{n+1} ; X_{n}^{(k)})
    = H(X_{n+1}) - H(X_{n+1} | X_{n}^{(k)})
    = H(X_{n+1}) - H(X_{n+1} , X_{n}^{(k)}) + H(X_{n}^{(k)})
    = H(X_{n+1}) + H(X_{n}^{(k)}) - H(X_{n+1}^{(k+1)})

    Args:
        x: symbolic sequence, symbols = [0, 1, ..., ns-1]
        kmax: history length
    Returns:
        ais: active information storage
    """

    n = len(x)
    h1 = H_k(x, ns, 1) # H(X[n+1])
    h2 = H_k(x, ns, k) # H(X[n:n-k])
    h3 = H_k(x, ns, k+1) # H(X[n+1], X[n:n-k])
    a = h1 + h2 - h3 # I(X[n+1]|X[n:n-k])
    return a


def aif(x, ns, kmax):
    """Time-lagged mutual information = Auto-information function (AIF).
    AIF of symbolic sequence x up to maximum lag kmax.
    *** Symbols must be 0, 1, 2, ... to use as indices ***
    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        kmax: maximum time lag
    Returns:
        aif_: time-lagged mutual information
    """

    #print("[+] AIF:")
    n = len(x)
    aif_ = np.zeros(kmax)
    for k in range(kmax):
        if (k%5 == 0): print("\t(aif)  k: {:d}".format(k))
        nmax = n-k
        h1 = H_1(x[:nmax], ns)
        h2 = H_1(x[k:k+nmax], ns)
        h12 = H_2(x[:nmax], x[k:k+nmax], ns)
        aif_[k] = h1 + h2 - h12
    return aif_/aif_[0]


def paif(x, ns, kmax):
    """Partial auto-information function (PAIF).
    PAIF of symbolic sequence x up to maximum lag kmax.
    *** Symbols must be 0, 1, 2, ... to use as indices ***
    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        kmax: maximum time lag
    Returns:
        paif_: partial autoinformation
    """

    #print("[+] PAIF:")
    n = len(x)
    aif_ = aif(x,ns,2)
    paif_ = np.zeros(kmax)
    paif_[0] = aif_[0]
    paif_[1] = aif_[1]
    for k in range(2,kmax):
        if (k%5 == 0): print("\t(paif) k: {:d}".format(k))
        h1 = H_k(x,ns,k)
        h2 = H_k(x,ns,k-1)
        h3 = H_k(x,ns,k+1)
        paif_[k] = 2*h1 - h2 - h3
    return paif_/paif_[0]


def excess_entropy_rate(x, ns, kmax, doplot=False):
    # y = ax+b: line fit to joint entropy for range of histories k
    # a = entropy rate (slope)
    # b = excess entropy (intersect.)
    h_ = np.zeros(kmax)
    for k in range(kmax):
        h_[k] = H_k(x, ns, k+1)
    ks = np.arange(1,kmax+1)
    a, b = np.polyfit(ks, h_, 1)
    # --- Figure ---
    if doplot:
        plt.figure(figsize=(6,6))
        plt.plot(ks, h_, '-sk')
        plt.plot(ks, a*ks+b, '-b')
        plt.tight_layout()
        plt.show()
    return (a, b)


def Tk_hat(x, ns, k=1):
    '''k-th order transition matrix.'''
    #print("[+] Estimating the transition matrix of order k = {:d}".format(k))
    L = ns**k # combinatorial number of k-histories
    pk = np.zeros(L) # distribution of histories
    Tk = np.zeros((L,ns)) # transition matrix from k-histories to next state
    sh = tuple(k*[ns]) # shape (ns...ns)
    N = len(x)-k
    # --- map histories to 1D indices ---
    d = {} # dictionary
    for i, idx in enumerate(np.ndindex(sh)): d[idx] = i
    for t in range(N):
        idx = tuple(x[t:t+k]) # p-history
        i = d[idx] # 1D index of p-history
        j = x[t+k] # symbol following the p-history idx
        pk[i] += 1.
        Tk[i,j] += 1.
    pk /= pk.sum()
    p_row = Tk.sum(axis=1, keepdims=True)
    p_row[p_row==0] = 1. # avoid division by zero
    Tk /= p_row # row sums --> 1.0
    return pk, Tk


def mc_k2(pk, Tk, k, ns, n):
    """k-th order Markov chain

    Args:
        pk: empirical k-dim symbol distribution (k-history, ns**k)
        Tk: empirical k-th order transition matrix (ns**k,ns)
        k: Markov order
        ns: number of symbols
        n: length of surrogate microstate sequence
    Returns:
        mc: surrogate Markov chain
    """

    # --- map histories to 1D indices ---
    sh = tuple(k*[ns]) # shape (ns...ns)
    d = {} # dictionary idx -> i
    dinv = np.zeros((ns**k,k))
    for i, idx in enumerate(np.ndindex(sh)):
        d[idx] = i
        dinv[i] = idx

    # --- NumPy vectorized code ---
    mc = np.zeros(n)
    pk_sum = np.cumsum(pk)
    mc[:k] = dinv[np.min(np.argwhere(np.random.rand() < pk_sum))]
    Tk_sum = np.cumsum(Tk, axis=1)
    for t in range(n-k-1):
        idx = tuple(mc[t:t+k]) # k-history
        i = d[idx] # 1D index of k-history
        j = np.min(np.argwhere(np.random.rand() < Tk_sum[i]))
        mc[t+k] = j

    return mc.astype('int')


def plot_aif_paif(x, ns, lmax):

    # --- AIF / PAIF ---
    print("[+] AIF:")
    aif_ = aif(x, ns, lmax)
    print("[+] PAIF:")
    paif_ = paif(x, ns, lmax)

    # --- Figure ---
    fs_tix = 14
    fs_label = 16
    fs_title = 18
    xyc = "axes fraction"
    ms_ = 5
    lw_ = 2
    lags = np.arange(lmax)
    z = np.zeros(lmax)
    ymax = 1.1

    fig, ax = plt.subplots(1, 2, figsize=(8,4)) # , sharex=True, sharey=True
    fig.patch.set_facecolor("white")

    # --- plot AIF ---
    ax[0].plot(lags, z, "-k", lw=1)
    ax[0].plot(lags, aif_, "ok", ms=ms_)
    for i in range(len(aif_)): ax[0].plot([i, i], [0, aif_[i]], "-k")
    xtix = [int(t) for t in ax[0].get_xticks()]
    ax[0].set_xticklabels(xtix, fontsize=fs_tix)
    ax[0].set_yticklabels(ax[0].get_yticks(), fontsize=fs_tix)
    ax[0].set_xlabel("lag", fontsize=fs_label)
    ax[0].set_ylabel(r"$\alpha_k$ [bits]", fontsize=fs_label)
    ax[0].set_ylim(-0.05, ymax)
    ax[0].set_title("AIF", fontsize=16, fontweight="bold")

    # --- plot PAIF ---
    ax[1].plot(lags, z, "-k", lw=1)
    ax[1].plot(lags, paif_, "ok", ms=ms_)
    for i in range(len(paif_)): ax[1].plot([i, i], [0, paif_[i]], "-k")
    xtix = [int(t) for t in ax[1].get_xticks()]
    ax[1].set_xticklabels(xtix, fontsize=fs_tix)
    ax[1].set_yticklabels(ax[1].get_yticks(), fontsize=fs_tix)
    ax[1].set_xlabel("lag", fontsize=fs_label)
    ax[1].set_ylabel(r"$\pi_{k}$ [bits]", fontsize=fs_label)
    ax[1].set_ylim(-0.05, ymax)
    ax[1].set_title("PAIF", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.show()


def main():
    f_mc13 = "mc_1_3.npy"
    f_mc3 = "mc_3.npy"
    f_even = "even.npy"
    f_gm2 = "gm2.npy"
    f_gm7 = "gm7.npy"
    f_Ising1 = "Ising2D_L50_N1000000_Temp2.27.npy"
    f_Ising2 = "Ising2D_L50_N1000000_Temp5.00.npy"
    f_ch = "ch.npy"
    f_ms = "eeg_ms.npy"
    x = np.load(f_mc13).astype('int')
    print(x.shape, x.dtype, set(x))
    ns = len(set(x))
    print("[+] Symbolic time series ns = {:d}".format(ns))

    plot_aif_paif(x, ns, kmax)
    # Markov surrogates
    for M in range(1,5):
        print("\n[+] Markov surrogates, order M = {:d}".format(M))
        pk, Tk = Tk_hat(x, ns, M)
        y = mc_k2(pk, Tk, M, ns, len(x))
        plot_aif_paif(y, ns, kmax)


if __name__ == "__main__":
    os.system("clear")
    N = int(1e5)
    kmax = 11
    alpha = 0.05
    ci = [100.*alpha/2., 100.*(1-alpha/2.)]
    # graphics
    fs_tix = 14
    fs_label = 16
    fs_title = 18
    xyc = "axes fraction"
    main()
