from rdkit import RDLogger
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RDLogger.DisableLog('rdApp.*')

latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 10
batch_size=32
noise_size=100

max_smile_len=128
min_smile_len=32

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

vocab_size=len(smiles_characters)

char2idx = {ch: i for i, ch in enumerate(smiles_characters)}
idx2char = {i: ch for i, ch in enumerate(smiles_characters)}