import numpy as np
import fastapi
from fastapi import File,UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from gensim.models import FastText
import esm
from typing import Optional
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])
import joblib
from DL_model import GatedAttention
from DL_utils import read_virus_seqs,visualize_attention_2d_heatmaps, test_one_virus
from DTI_model import DrugTargetGNN
from DTI_utils import read_virus,test_antivirus,test_top_antivirus
from ML_utils import Virus2Vec, get_predection_per_virus
from GAN_utils import get_new_antivirus
from GAN_model import Generator


# CONSTANTS
SG_EMBEDD_SIZE=30
SG_WINDOW=5
# Transformer Parameters
N_HEAD = 5         # Number of attention heads
ENCODER_N_LAYERS = 2       # Number of transformer layers
EMBEDDING_SIZE=SG_EMBEDD_SIZE
INTERMIDIATE_DIM=512
smiles_characters = [
    'Cl', 'Br','Na','Si','Mg','Zn','Se','Te','se','As','te','Ag','Al',
    'B','C', 'N', 'O', 'P', 'S', 'F', 'I','K',
    'b', 'c', 'n', 'o', 'p', 's',
    '-', '=', '#', ':', '/', '\\',
    '(', ')',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '[', ']',
    '+', '-', '@', '@@', '.', '%',
    'H', 'h'  ,
    '<EOS>',
    '<SOS>',
    '<PAD>'
]
char2idx = {ch: i for i, ch in enumerate(smiles_characters)}
idx2char = {i: ch for i, ch in enumerate(smiles_characters)}
vocab_size = len(smiles_characters)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DL_model = GatedAttention(N_HEAD, ENCODER_N_LAYERS, EMBEDDING_SIZE, INTERMIDIATE_DIM).to(device)
DL_model.load_state_dict(torch.load(r"models\model_weights.pth", map_location=device))
ft_model = FastText.load(r"models\ft_skipgram.model")
antivirus_model = DrugTargetGNN()
antivirus_model.load_state_dict(torch.load(r"models\drug_target_model2.pth", map_location=device))
esm_model, esm_alphabet=esm.pretrained.load_model_and_alphabet_local(r"models\esm1b_t33_650M_UR50S.pt")
esm_model.eval()
Virus2Vec_model=joblib.load(r"models\RF_Virus2vec.pkl")
checkpoint_path=r"models\checkpoint_epoch_38.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
generator = Generator(vocab_size).to(device)
generator.load_state_dict(checkpoint['generator_state_dict'])



app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-host-ML")
async def predict_host_ML(file: Optional[UploadFile] = File(None),
    virus: Optional[str] = Form(None)):
    # Case 1: File is uploaded (FASTA)
    if file:
        content = (await file.read()).decode()
        virus = read_virus_seqs(content)
    # Case 2: Use virus sequence from form field
    elif virus:
        virus = read_virus_seqs(virus)

    datas = virus["Sequence"]
    Virus2Vec_feature_vector = Virus2Vec(datas)

    Y_prob = get_predection_per_virus(Virus2Vec_model,Virus2Vec_feature_vector)
    print(f"Y_prob: {Y_prob}")
    return {
        "probability": Y_prob,
    }

@app.post("/predict-host")
async def predict_host(file: Optional[UploadFile] = File(None),
    virus: Optional[str] = Form(None)):
        # Case 1: File is uploaded (FASTA)
    if file:
        content = (await file.read()).decode()
        virus = read_virus_seqs(content)
    # Case 2: Use virus sequence from form field
    elif virus:
        virus = read_virus_seqs(virus)

    ids_original = virus["Virus_ID"]
    datas = virus["Sequence"]
    seq_ids_original = virus["Seq_ID"]

    _, ids = np.unique(ids_original, return_inverse=True)
    _, seq_ids = np.unique(seq_ids_original, return_inverse=True)

    Y_prob, Y_hat, A, A_2 = test_one_virus(DL_model,ft_model,datas,ids,seq_ids)

    A = [a.cpu().numpy() if isinstance(a, torch.Tensor) and a.is_cuda else a.numpy() if isinstance(a, torch.Tensor) else np.array(a) for a in A]
    A_2 = [a_2.cpu().numpy() if isinstance(a_2, torch.Tensor) and a_2.is_cuda else a_2.numpy() if isinstance(a_2, torch.Tensor) else np.array(a_2) for a_2 in A_2]
    A = np.array(A)
    A_2 = np.array(A_2)      
    base64_img = visualize_attention_2d_heatmaps(A, A_2, seq_ids_original, ids_original)
    return {
        "probability": Y_prob.tolist()[0],
        "img":base64_img
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

@app.post("/generate-antivirus")
async def generate_antivirus(
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
        
    drugs=get_new_antivirus(generator,antivirus_model, esm_model, esm_alphabet, virus_seq, char2idx, idx2char)
    return {
        "drugs":drugs
    }