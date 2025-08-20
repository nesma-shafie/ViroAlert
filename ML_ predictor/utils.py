import os
import pandas as pd
import numpy as np
from itertools import product
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib

# read dataset
file_name_ncbi_datas = "/kaggle/input/ncbi-data-csv/ncbi_cleaned_train_data.csv"
file_name_gasaid_datas = "/kaggle/input/ncbi-data-csv/gasaid_cleaned_train_data.csv"
ncbi_test_file_name = "/kaggle/input/ncbi-data-csv/ncbi_cleaned_test_data.csv"
gasaid_test_file_name = "/kaggle/input/ncbi-data-csv/gasaid_cleaned_test_data.csv"
covid_test = "/kaggle/input/ncbi-data-csv/test_covid.csv"
cows_test = "/kaggle/input/ncbi-data-csv/Cows.csv"

Asian_flu_test = "/kaggle/input/testing-pandemics-csv/Asian Flu (1957-1958).csv"
hong_kong_flu_test = "/kaggle/input/testing-pandemics-csv/Hong Kong (1968-1970).csv"
spanish_flu_test = "/kaggle/input/testing-pandemics-csv/Spanish Flu (1918-1920).csv"
pdmh1n1_flu_test = "/kaggle/input/testing-pandemics-csv/Swine Flu (2009-2010).csv"

# Parameters
k = 3  # k-mer size

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


def read_data_from_file(filename):
    file_path = os.path.abspath(filename)

    # Read CSV file
    df = pd.read_csv(file_path)

    # Rename columns
    df.columns = ["ID", "Sequence"]

    # Drop rows where Sequence is not a string or is missing
    df = df[df["Sequence"].apply(lambda x: isinstance(x, str))].copy()
    df = df[
        df["Sequence"].apply(lambda x: len(x) > 200 if isinstance(x, str) else False)
    ].copy()
    df = df[~df["Sequence"].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]")]

    # Extract fields
    df["Class"] = df["ID"].apply(lambda x: x.split("|")[-1] if "|" in x else "")

    return df[["Sequence", "Class"]]


def remove_common_sequences(df1, df2, column="Sequence"):
    common_mask = df1[column].isin(df2[column])
    if common_mask.any():
        count = common_mask.sum()
        print(f"{count} common sequences found. Removing them from df2...")
        df1_cleaned = df1[~common_mask].copy()
    else:
        print("No common sequences found between the two DataFrames.")
        df1_cleaned = df1.copy()
    return df1_cleaned


def remove_duplicates(df, subset=None, show_duplicates=True):
    duplicated_mask = df.duplicated(subset=subset)
    has_duplicates = duplicated_mask.any()

    if has_duplicates:
        num_duplicates = duplicated_mask.sum()
        print(f"Found {num_duplicates} duplicated row(s). Removing them...")
        df = df.drop_duplicates(subset=subset)
    else:
        print("No duplicated rows found.")
    return df


def get_all_combinations(alphabet, k):
    return ["".join(p) for p in product(alphabet, repeat=k)]


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
    ppm = comp_ppm(pfm)
    # Step 3: Compute log-odds PWM
    return np.log2(ppm / p_c[:, np.newaxis])


def comp_mmers_score(mmer, pwm):
    # Score is sum of weights for each character position
    # W("AC") = W['A'][0] + W['C'][1]
    return sum(pwm[char_to_idx[c], i] for i, c in enumerate(mmer))


def cls_report(y_test, preds):
    print(classification_report(y_test, preds, zero_division=0))
    print("Accuracy:", accuracy_score(y_test, preds))


# Random Forest
def RandomForest(X_test, y_test, model):
    preds = model.predict(X_test)
    print("🎯 Random Forest Results:")
    cls_report(y_test, preds)
    return preds


# Logistic Regression
def LogisticReg(X_test, y_test, model):
    preds = model.predict(X_test)
    print("🎯 Logistic Regression Results:")
    cls_report(y_test, preds)
    return preds


# Support Vector Machine
def SVM(X_test, y_test, model):
    preds = model.predict(X_test)
    print("🎯 Support Vector Machine Results:")
    cls_report(y_test, preds)
    return preds


def con_matrix(y_test, preds):
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, preds)

    # Define display labels (0 = human, 1 = animal)

    # Create and display the confusion matrix with labels
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Human", "Animal"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def plot_tsne(X, y):
    X_pca = PCA(n_components=80).fit_transform(X)
    X_embedded = TSNE(n_components=2).fit_transform(X_pca)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=["blue" if label == 0 else "red" for label in y],
        alpha=0.5,
    )

    # Add legend
    for label, color, name in zip([0, 1], ["blue", "red"], ["Human", "Non-Human"]):
        plt.scatter([], [], c=color, label=name)
    plt.legend()

    plt.title("t-SNE Visualization of Virus2Vec Embeddings")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.show()
