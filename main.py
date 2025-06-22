import numpy as np
import fastapi
import pandas as pd
from fastapi import File,UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from gensim.models import FastText
from model import GatedAttention,DrugTargetGNN
import esm
from typing import Optional
import pickle
from utils import read_virus,read_virus_seqs,test_one_virus,test_antivirus,test_top_antivirus
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])


# CONSTANTS
SG_EMBEDD_SIZE=30
SG_WINDOW=5
# Transformer Parameters
N_HEAD = 5         # Number of attention heads
ENCODER_N_LAYERS = 2       # Number of transformer layers
EMBEDDING_SIZE=SG_EMBEDD_SIZE
INTERMIDIATE_DIM=512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GatedAttention(N_HEAD, ENCODER_N_LAYERS, EMBEDDING_SIZE, INTERMIDIATE_DIM).to(device)
model.load_state_dict(torch.load(r"models\model_weights.pth", map_location=device))
ft_model = FastText.load(r"models\ft_skipgram.model")
antivirus_model = DrugTargetGNN()
antivirus_model.load_state_dict(torch.load(r"models\drug_target_model.pth", map_location=device))
esm_model, esm_alphabet=esm.pretrained.load_model_and_alphabet_local(r"models\esm1b_t33_650M_UR50S.pt")
esm_model.eval()



app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-host")
async def predict_host(file: UploadFile=File(...)):
    
    content = (await file.read()).decode()
    # Read the virus data
    virus = read_virus_seqs(content)

    ids_original = virus["Virus_ID"]
    datas = virus["Sequence"]
    seq_ids_original = virus["Seq_ID"]

    _, ids = np.unique(ids_original, return_inverse=True)
    _, seq_ids = np.unique(seq_ids_original, return_inverse=True)

    #  Make prediction
    Y_prob, Y_hat, A, A_2 = test_one_virus(model,ft_model,datas,ids,seq_ids)
    print(f"Y_prob: {Y_prob}, Y_hat: {Y_hat}")

    return {
        "prediction": Y_hat.tolist()[0],
        "probability": Y_prob.tolist()[0],
    }
@app.post("/predict-antivirus")
async def predict_antivirus(
    file: Optional[UploadFile] = File(None),
    virus: Optional[str] = Form(None),
    smiles: Optional[str] = Form(...)
):
    # Case 1: File is uploaded (FASTA)
    if file:
        content = (await file.read()).decode()
        virus_seq = read_virus(content)
    # Case 2: Use virus sequence from form field
    elif virus:
        virus_seq = virus
    pIC50 = test_antivirus(antivirus_model, esm_model, esm_alphabet, virus_seq, smiles)
    return {
        "pIC50": pIC50.item(),
    }

@app.post("/top-antivirus")
async def top_antivirus(
    file: Optional[UploadFile] = File(None),
    virus: Optional[str] = Form(None),
):
    # Case 1: File is uploaded (FASTA)
    if file:
        content = (await file.read()).decode()
        virus_seq = read_virus(content)
    # Case 2: Use virus sequence from form field
    elif virus:
        virus_seq = virus
    top_smiles = test_top_antivirus(antivirus_model, esm_model, esm_alphabet, virus_seq)
    return {
        "top_smiles": top_smiles,
    }