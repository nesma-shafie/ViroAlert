import numpy as np
from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import gc
import joblib
from joblib import Parallel, delayed
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os
import itertools
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import itertools
import seaborn as sns


# Parameters
k = 4  # k-mer size
m = 3  # minimizer size
pseudo_count = 0.1
alphabet = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
alphabet = sorted(set(alphabet))
alphabet_size = len(alphabet)
char_to_idx = {c: i for i, c in enumerate(alphabet)}


def get_all_combinations(alphabet, k):
    return ["".join(p) for p in product(alphabet, repeat=k)]


# All the m-mers (not k-mers!) combinations
combos = get_all_combinations(alphabet, m)
combo_index = {mer: i for i, mer in enumerate(combos)}  # Faster lookup


def get_kmers(seq, k):
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def get_minimizer(kmer, m):
    """
    Get lex smallest m-mer in both the k-mer and its reverse.
    """
    kmer_rev = kmer[::-1]
    all_mmers = [kmer[i : i + m] for i in range(len(kmer) - m + 1)]
    all_mmers += [kmer_rev[i : i + m] for i in range(len(kmer_rev) - m + 1)]
    return min(all_mmers)


def comp_minimizers(seq, k, m):
    """Return list of minimizers for k-mers in sequence"""
    kmers = get_kmers(seq, k)
    return [get_minimizer(kmer, m) for kmer in kmers]


def get_alphabet_count(col, alphabet):
    """Return count vector of characters in one column of matrix A"""
    counter = Counter(col)
    return np.array([counter.get(char, 0) for char in alphabet])


def get_p_c():
    # Codon-based number of mappings to amino acids
    codon_table = {
        "A": 4,
        "C": 2,
        "D": 2,
        "E": 2,
        "F": 2,
        "G": 4,
        "H": 2,
        "I": 3,
        "K": 2,
        "L": 6,
        "M": 1,
        "N": 2,
        "P": 4,
        "Q": 2,
        "R": 6,
        "S": 6,
        "T": 4,
        "V": 4,
        "W": 1,
        "Y": 2,
    }
    # Return vector for p(c) aligned with the alphabet
    return np.array([codon_table.get(c, 1) / 61 for c in alphabet])


def comp_ppm(pfm):
    # Step 1: Add pseudocounts
    pfm = pfm + pseudo_count
    # Step 2: Compute PPM with small constant to avoid division by zero
    return pfm / (np.sum(pfm, axis=0) + 1e-9)


def comp_pwm(pfm, p_c):
    # Step 1: Add pseudocounts
    pfm = pfm + pseudo_count
    # Step 2: Compute PPM with small constant to avoid division by zero
    ppm = pfm / (np.sum(pfm, axis=0) + 1e-9)
    # Step 3: Compute log-odds PWM
    return np.log2(ppm / p_c[:, np.newaxis])


def comp_mmers_score(mmer, pwm):
    # Score is sum of weights for each character position
    # W("AC") = W['A'][0] + W['C'][1]
    return sum(pwm[char_to_idx[c], i] for i, c in enumerate(mmer))


pfm = np.zeros((alphabet_size, m))
v = np.zeros(len(combos))


def get_predection_per_virus(model, X_test, Verbose=False):
    preds = model.predict_proba(X_test)
    return np.mean(preds[:, 0])


def Virus2Vec(S, alphabet=alphabet, k=k, m=m):
    V = np.zeros((len(S), len(combos)), dtype=np.float32)
    for j, seq in enumerate(tqdm(S, desc="Processing sequences")):
        A = comp_minimizers(seq, k, m)  # List of minimizers (m-mers)

        pfm.fill(0)
        for i in range(m):
            # the chars in pos i in A
            col = [a[i] for a in A]
            # PFM[c][i] = count of character c at position i across all m-mers
            pfm[:, i] = get_alphabet_count(col, alphabet)

        p_c = get_p_c()
        pwm = comp_pwm(pfm, p_c)

        # Step 4: Score for all Minimizers
        W = [comp_mmers_score(mmer, pwm) for mmer in A]
        # Step 5
        v.fill(0)  # feature vector of size |Σ|^m
        for i in range(len(A)):  # A contains minimizers (m-mers)
            idx = combo_index[A[i]]  # find index of the i-th m-mer
            v[idx] += W[i]  # add the minimizer's score
        V[j, :] = v  # Assign the computed vector directly to row j
    return V
