import numpy as np
import fastapi
import os
from fastapi import File,UploadFile,Query
from fastapi.middleware.cors import CORSMiddleware
import torch
from gensim.models import FastText
from model import GatedAttention,DrugTargetGNN
import esm
from esm.pretrained import load_model_and_alphabet_core

from torch.serialization import add_safe_globals
from utils import read_data_from_file,test_one_virus, test_antivirus
import argparse
from torch.serialization import add_safe_globals

# Allow safe unpickling of argparse.Namespace
# add_safe_globals([argparse.Namespace])


# CONSTANTS
UPLOAD_DIR = "tmp"
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
model_data = torch.load(r"models\esm1b_t33_650M_UR50S.pt", map_location="cpu")
# esm_model, esm_alphabet = load_model_and_alphabet_core(model_data)
# esm_model.eval()


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
    os.makedirs(UPLOAD_DIR, exist_ok=True)  # create if not exists
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    

    # Read the virus data
    virus = read_data_from_file(temp_path)

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
# @app.post("/predict-antivirus")
# async def predict_antivirus(virus: str = Query(...), smiles: str = Query(...)):
#     # Predict using the test_antivirus function
#     pIC50 = test_antivirus(antivirus_model, esm_model, esm_alphabet, virus, smiles)
    
#     return {
#         "pIC50": pIC50
#     }