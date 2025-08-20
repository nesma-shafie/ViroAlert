
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from CONSTANTS import char2idx, max_smile_len, smiles_characters
import re
import torch
def tokenize_smiles(smiles):
    token_pattern = re.compile("^(" + "|".join(re.escape(token) for token in smiles_characters) + ")+$")
    token_regex = re.compile("(" + "|".join(re.escape(token) for token in smiles_characters) + ")")
    if not token_pattern.match(smiles):
        print(smiles)
        print(token_regex.findall(smiles))
        return False  
    tokens= token_regex.findall(smiles)
    tokens=["<SOS>"]+tokens+["<EOS>"]
    return tokens

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, char2idx):
        self.smiles_list = smiles_list
        self.char2idx = char2idx

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        seq = self.smiles_list[idx]
        tokenized = [self.char2idx[c] for c in tokenize_smiles(seq)] 
        return torch.tensor(tokenized, dtype=torch.long)

def collate_fn(batch):
    pad_len = max_smile_len - batch[0].size(0)
    padding = torch.full((pad_len,), char2idx["<PAD>"], dtype=batch[0].dtype, device=batch[0].device)
    batch[0] = torch.cat([batch[0], padding], dim=0)
    return pad_sequence(batch, batch_first=True, padding_value=char2idx["<PAD>"])