

import torch
import torch.optim as optim
import gc
from CONSTANTS import device
import tqdm
import esm

from DTI import DrugTargetGNN, protein_graph, protein_graph_to_data, smile_graph, drug_graph_to_data
from CONSTANTS import char2idx
def get_top_anti_viruses(virus,model,esm_model,esm_alphabet,smiles):
    virus_graph=protein_graph(esm_model,esm_alphabet,virus)
    virus_graph = protein_graph_to_data(virus_graph).to(device)
    inhibitors=[]
    model.eval()
    th=7
    with torch.no_grad():
        for smile in tqdm(smiles , desc="get_top_anti_viruses"):
            drug_graph=smile_graph(smile)
            drug_graph = drug_graph_to_data(drug_graph).to(device)
            output = model(virus_graph, drug_graph)
            if output>th:
               inhibitors.append(smile)
    print("inhibitors",len(inhibitors))
    return inhibitors 


def train_for_generator_fine_tunnning(generator, discriminator, inhibitors, epochs=3, max_len=128):
    print("Starting fine-tuning...")
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
    g_scaler = torch.amp.GradScaler('cuda')  # Updated for PyTorch 2.4+    
    g_steps = 1

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for _ in range(g_steps):
            g_optimizer.zero_grad()
            loss_g = generator.train_step(char2idx['<SOS>'], max_len, g_optimizer, None,g_scaler,fine_tune_flag=True,inhibitors=inhibitors)
           
            print(f"Epoch {epoch}, Generator Loss: {loss_g:.4f}")
        torch.cuda.empty_cache()  # Clear memory after each epoch
        gc.collect()
def fine_tune_on_virus(virus,drugs,generator, discriminator):
    DTI_model = DrugTargetGNN()
    DTI_model.load_state_dict(torch.load(r"/kaggle/input/drug-target/pytorch/default/2/drug_target_model2.pth", map_location=device))
    DTI_model.to(device)
    esm_model, esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm_model.eval()
    
    inhibitors=get_top_anti_viruses(virus,DTI_model,esm_model, esm_alphabet,drugs)
    train_for_generator_fine_tunnning(generator, discriminator, inhibitors)
