#!/usr/bin/env python

import math
import pickle
import time
from itertools import combinations, permutations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from olympus.surfaces import CatCamel, CatDejong, CatMichalewicz, CatSlope
from scipy.special import binom


# ------------------
# HELPER FUNCTIONS
# ------------------
def stirling_sum(Ns):
    """..."""
    stirling = lambda n, k: int(
        1.0
        / math.factorial(k)
        * np.sum([(-1.0) ** i * binom(k, i) * (k - i) ** n for i in range(k)])
    )
    return np.sum([stirling(Ns, k) for k in range(Ns + 1)])


def partition(S):
    """..."""
    if len(S) == 1:
        yield [S]
        return

    first = S[0]
    for smaller in partition(S[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        yield [[first]] + smaller


def gen_partitions(S):
    """
    generate all possible partitions of Ns-element set S

    Args:
        S (list): list of non-functional parameters S
    """
    return [p for _, p in enumerate(partition(S), 1)]


def gen_permutations(X_funcs, Ng):
    """generate all possible functional parameter permutations
    given number of non-functional parameter subsets Ng

    Args:
        X_funcs (np.ndarray): numpy array with all functional
            possile functional parameters
        Ng (int): number of non-functional parameter subsets

    Returns
        (np.ndarray): array of parameter permutations of
            shape (# perms, Ng, # params)
    """

    return np.array(list(permutations(X_funcs, Ng)))


def measure_objective(xgs, G, surf_map):
    """..."""
    f_x = 0.0
    for g_ix, Sg in enumerate(G):
        f_xg = 0.0
        for si in Sg:
            f_xg += measure_single_obj(xgs[g_ix], si, surf_map)
        f_x += f_xg / len(Sg)

    return f_x


def record_merits(S, surf_map, X_func_truncate=20):

    # list of dictionaries to store G, X_func, f_x
    f_xs = []

    start_time = time.time()

    # generate all the partitions of non-functional parameters
    Gs = gen_partitions(S)
    print("total non-functional partitions : ", len(Gs))

    # generate all the possible values of functional parametres
    param_opts = [f"x{i}" for i in range(21)]
    cart_product = list(product(*param_opts))
    X_func = np.array([list(elem) for elem in cart_product])

    if isinstance(X_func_truncate, int):
        X_funcs = X_funcs[:X_func_truncate, :]
    print("cardnality of functional params : ", X_funcs.shape[0])

    for G_ix, G in enumerate(Gs):
        if G_ix % 1 == 0:
            print(f"[INFO] Evaluating partition {G_ix+1}/{len(Gs)+1}")
        Ng = len(G)
        # generate permutations of functional params
        X_func_perms = gen_permutations(X_funcs, Ng)

        for X_func in X_func_perms:
            # measure objective
            f_x = measure_objective(X_func, G, surf_map)
            # store values
            f_xs.append(
                {
                    "G": G,
                    "X_func": X_func,
                    "f_x": f_x,
                }
            )
    total_time = round(time.time() - start_time, 2)
    print(f"[INFO] Done in {total_time} s")

    return f_xs


if __name__ == "__main__":

    # -------------
    # TOY PROBLEM
    # -------------
    # S = [0, 1, 2, 3] # four non-functional parameter options
    S = [0, 1, 2]
    surf_map = {
        0: CatCamel(param_dim=2, num_opts=21),
        1: CatDejong(param_dim=2, num_opts=21),
        2: CatMichalewicz(param_dim=2, num_opts=21),
        3: CatSlope(param_dim=2, num_opts=21),
    }

    # f_xs = record_merits(S, surf_map, X_func_truncate=20)

    param_opts = [f"x{i}" for i in range(21)]
    cart_product = list(product(*param_opts))
    X_func = np.array([list(elem) for elem in cart_product])

    print(X_func.shape)
