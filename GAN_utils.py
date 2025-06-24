from rdkit import Chem
from rdkit.Chem import SanitizeMol
from rdkit.Chem import AllChem
from rdkit import DataStructs
import torch.optim as optim
import numpy as np
import torch
import gc
import pandas as pd
import pickle
from model import DrugTargetGNN
import esm
from model import  protein_graph_to_data, drug_graph_to_data
from utils import smile_graph,protein_graph
from tqdm import tqdm
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# from ViroGen.database_conn import conn
from model import DrugProteinDataset, custom_collate


def get_validity_metric(smiles):
    try:
        if smiles[0] == '<':
            smiles = smiles[5:]
        if smiles[-1] == '>':
            smiles = smiles[:-5]
            
        mol = Chem.MolFromSmiles(smiles)#remove <SOS> from begin
        if mol is None:
            return 0
        SanitizeMol(mol)
        return 1
    except Exception:
        return 0

def get_fingerprint(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def similarity_score(smiles,inhibitors):
    inhibitor_fps = [get_fingerprint(s) for s in inhibitors]
    smile_fps = [get_fingerprint(s) for s in smiles]
    
    total_max_scores = []
    for smile_fp in smile_fps:
        similarities = DataStructs.BulkTanimotoSimilarity(smile_fp, inhibitor_fps)
        total_max_scores.append(max(similarities))  # Or use np.mean(similarities) if you want
    return np.mean(total_max_scores)

def get_top_anti_viruses(virus,model,esm_model,esm_alphabet):
    max_smile_len=128
    min_smile_len=32
    th=7
    with open(r"data/drug_graphs.pkl", "rb") as f:
        drug_graphs = pickle.load(f)
    drugs=pd.read_csv(r"data/drugs.csv")
    drugs = drugs['SMILES']
    drugs = drugs[drugs.str.len().between(min_smile_len, max_smile_len-2)]
    drugs = drugs.tolist()

    virus_graph = protein_graph(esm_model, esm_alphabet, virus)
    virus_graph = [virus_graph] * len(drug_graphs)
    dataset= DrugProteinDataset(virus_graph, drug_graphs)
    loader=DataLoader(dataset, batch_size=64, collate_fn=custom_collate)
    model.eval()
    outputs=[]
    for protein_graphs, drug_graphs in loader:
            protein_graphs = protein_graphs.to(device)
            drug_graphs = drug_graphs.to(device)
            with torch.no_grad():
                output = model(protein_graphs, drug_graphs)
            outputs.extend(output.cpu().detach().numpy())
    outputs = np.array(outputs)
    top_indices = np.where(outputs > th)[0]
    inhibitors=[drugs[i] for i in top_indices]
    
    print(" top ",len(inhibitors)," inhibitors found with threshold ",th)
    return inhibitors 

def train_for_generator_fine_tunnning(generator, inhibitors, char2idx, idx2char, epochs=50, max_len=128):
    print("Starting fine-tuning...")
    batch_size=32
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
    g_steps = 1

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for _ in range(g_steps):
            g_optimizer.zero_grad()

            # Sample a molecule from the generator
            generated, _ = generator.sample(char2idx['<SOS>'], max_len,batch_size)
            smiles=np.array([])
            for i in range(generated.size(0)):
                smile=''.join([idx2char[t] for t in generated[i].tolist() if t !=char2idx['<SOS>'] and t !=char2idx['<EOS>'] and t !=char2idx['<PAD>'] ])
                # Check and process validity
                if get_validity_metric(smile):
                    smiles=np.append(smiles,smile )
            
            if len(smiles)>0:

                # Calculate reward
                validity = len(smiles)/batch_size
                sim_score = similarity_score(smiles, inhibitors)
                #try without disc
                reward = 0.5 * validity  + 0.5 * sim_score
                loss = -reward
                print(f"Loss: {loss:.4f}")

                loss = torch.tensor(loss, requires_grad=True, device=generated.device)
                loss.backward()
                g_optimizer.step()

            else:
                print("Invalid molecule — skipping update.")

        torch.cuda.empty_cache()
        gc.collect()

def fine_tune_on_virus(generator,DTI_model, esm_model, esm_alphabet,virus, char2idx, idx2char):
    inhibitors=get_top_anti_viruses(virus,DTI_model,esm_model, esm_alphabet)
    train_for_generator_fine_tunnning(generator, inhibitors, char2idx, idx2char)

def get_new_antivirus(generator,DTI_model, esm_model, esm_alphabet, new_virus, char2idx, idx2char): 
    fine_tune_on_virus(generator,DTI_model, esm_model, esm_alphabet,new_virus, char2idx, idx2char)
    
    best_drugs = []
    drug_7=0
    trials=0
    virus_graph=protein_graph(esm_model,esm_alphabet,new_virus)
    virus_graph = protein_graph_to_data(virus_graph).to(device)
    with torch.no_grad():
        while(len(best_drugs)<5 and trials<5):
            print("trial ",trials)
            n=100
            trials+=1
            generated, _ = generator.sample(char2idx['<SOS>'], 128,n,temperature=1)
            for i in range(n):
                smiles = ''.join([idx2char[t] for t in generated[i].tolist() if t !=char2idx['<SOS>'] and t !=char2idx['<EOS>'] and t !=char2idx['<PAD>'] ])
                valid=get_validity_metric(smiles)
                if valid:
                    drug_graph=smile_graph(smiles)
                    drug_graph = drug_graph_to_data(drug_graph).to(device)
                    
                    output = DTI_model(virus_graph, drug_graph)
                    # print("out",output,"valid",valid)
                    if(output.item()>7):
                        drug_7+=1
                        best_drugs.append((smiles,output.item()))
                        
    return best_drugs         
