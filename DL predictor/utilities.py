import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import CONSTANTS
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from MIL_Dataset import MILDataset, collate_fn
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
from Bio import SeqIO
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ASW(sequence, l_sub):
   
    l = len(sequence)
    if CONSTANTS.N> 1:
        l_stride = (l - l_sub) // (CONSTANTS.N - 1)
    else:
        l_stride = 1  
    subsequences = []
    for i in range(0, min(CONSTANTS.N * l_stride, l - l_sub + 1), l_stride):
        subsequences.append(sequence[i:i + l_sub])
    return subsequences

def get_lsub_sequence(df):        
    # get the length of the longest seq
    llongest=max(df['Length'])
    lshortest=min(df['Length'])
    print("llongest",llongest)
    llongest=max(df['Length'])
    print("lshortest",lshortest)
    lower_bound = int(llongest / CONSTANTS.N)
    upper_bound = int(llongest - CONSTANTS.N + 1)
    l_sub_array=np.arange(lower_bound, upper_bound + 1)
    l_sub=lshortest-CONSTANTS.N+1
    if l_sub not in l_sub_array:
        print("error ASW")
    CONSTANTS.l_sub= l_sub
    return 


def tokenize(datas,ids,seq_ids,labels,ft_model):
    datas = [ASW(sequence,CONSTANTS.l_sub) for sequence in datas.tolist()]
    labels= np.repeat(labels, CONSTANTS.N).tolist()
    ids=np.repeat(ids, CONSTANTS.N).tolist()
    seq_ids=np.repeat(seq_ids, CONSTANTS.N).tolist()
    
    # Apply FastText (CBOW)
    keys_wv=set(list(ft_model.wv.key_to_index.keys()))
    
    embeddings = np.array([
        ft_model.wv[k]  
        for kmer in tqdm(datas, desc="FastText inference")
        for k in kmer
    ])
    embeddings=torch.tensor(embeddings).to(device)
    ids=torch.tensor(ids).to(device)
    seq_ids=torch.tensor(seq_ids).to(device)
    labels=torch.tensor(labels).to(device)
    return embeddings,ids,seq_ids,labels

def create_data_loader(datas,labels,ids,seq_ids,ft_model):
    embeddings, ids,seq_ids, labels=tokenize(datas,ids,seq_ids,labels,ft_model)
    mildataset = MILDataset(embeddings, ids,seq_ids, labels)
    data_loader = DataLoader(mildataset, batch_size=CONSTANTS.BATCH_SIZE, shuffle=True,num_workers=0, collate_fn=collate_fn)
    return data_loader


def read_data_from_file(filenames):
    dfs = []  

    for filename in filenames:
        file_path = os.path.abspath(filename)  
        df = pd.DataFrame.from_records([
            {
                "ID": "|".join(record.description.split("|")[1:]),
                "Sequence": str(record.seq),  
                "year":record.description.split("|")[-2]
            }
            for record in SeqIO.parse(file_path, "fasta")
        ])
          
        dfs.append(df)  
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def read_data_from_csv(filename):
    file_path = os.path.abspath(filename)
    df = pd.read_csv(file_path)
    df.columns = ["ID", "Sequence"]
    df = df[df["Sequence"].apply(lambda x: isinstance(x, str))].copy()
    df = df[df["Sequence"].apply(lambda x: len(x) > 200 if isinstance(x, str) else False)].copy()

    df["Virus_ID"] = df["ID"].apply(lambda x: "".join(x.split("|")[1:]) if "|" in x else "")
    df["Seq_ID"] = df["ID"].apply(lambda x: x.split("|")[0] if "|" in x else "")
    df["Class"] = df["ID"].apply(lambda x: x.split("|")[-1] if "|" in x else "")
    df["Length"] = df["Sequence"].apply(len)
    return df[["Sequence", "Virus_ID", "Seq_ID", "Class", "Length"]]
