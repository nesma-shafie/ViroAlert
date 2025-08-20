import torch
from torch.utils.data import Dataset
from tqdm import tqdm



class MILDataset(Dataset):
    def __init__(self, datas, ids, seq_ids, labels):
        self.datas = datas  # Instance features
        self.ids = ids # Virus (outer bag) IDs
        self.seq_ids = seq_ids  # Sequence (inner bag) IDs
        self.labels = labels.to("cpu")  # Labels at the virus (outer bag) level

        # Unique IDs for outer bags (viruses) and their indices
        self.unique_virus_ids, self.virus_indices = torch.unique(self.ids, return_inverse=True)
        
        # Unique IDs for inner bags (sequences) and their indices
        self.unique_seq_ids, self.seq_indices = torch.unique(self.seq_ids, return_inverse=True)

        # Mapping from virus to instance indices  2d array each list is the virus data indecies
        self.virus_bag_indices_list = [torch.where(self.virus_indices == i)[0].to("cpu") for i in tqdm(range(len(self.unique_virus_ids)))]

        # Mapping from sequence to instance indices 2d array each list is the seq data indecies
        self.seq_bag_indices_list = [torch.where(self.seq_indices == i)[0].to("cpu") for i in tqdm( range(len(self.unique_seq_ids)))]

        # Labels assigned at the virus level (each virus gets one label)
        self.virus_labels = [self.labels[indices[0]] for indices in self.virus_bag_indices_list]

        # Precomputed bag-of-bags structure (virus → [seq])
        self.virus_seq_map = {}  # Maps virus_id -> list of sequence indices
        for i, virus_id in tqdm(enumerate(self.unique_virus_ids)):
            self.virus_seq_map[virus_id.item()] = list((self.seq_ids[self.virus_bag_indices_list[i]].tolist()))

        # Precomputed bag IDs for each virus and sequence
        self.precomputed_virus_ids = [torch.full((indices.shape[0],), self.unique_virus_ids[i], dtype=torch.long) 
                                    for i, indices in enumerate(self.virus_bag_indices_list)]

        self.datas = self.datas.cpu()


    def __len__(self):
        return len(self.unique_virus_ids)  # Number of unique viruses (outer bags)

    def __getitem__(self, index):
        
        # Get all instance indices belonging to this virus
        virus_instance_indices = self.virus_bag_indices_list[index]
        # Retrieve instance-level data
        virus_data = self.datas[virus_instance_indices]
        virus_label = self.virus_labels[index]
        virus_id = self.precomputed_virus_ids[index]
        # Find which sequences belong to this virus
       
        seq_ids_in_virus = self.virus_seq_map[virus_id[0].item()]

        return {
            "virus_id": virus_id,
            "virus_data": virus_data,
            "virus_label": virus_label,
            "seq_id": seq_ids_in_virus
        }


def collate_fn(batch):


    all_virus_ids = []
    all_virus_data = []
    all_virus_labels = []
    all_virus_seq_ids = []
   

    for item in batch:
        virus_id = item["virus_id"].tolist()
        virus_data = item["virus_data"].tolist()
        virus_label = item["virus_label"]
        seq_id = item["seq_id"]

        all_virus_seq_ids.extend(seq_id)
        all_virus_ids.extend(virus_id)
        all_virus_data.extend(virus_data)
        all_virus_labels.append(virus_label)
    
    batch_virus_labels = torch.tensor(all_virus_labels, dtype=torch.float)
    batch_seq_ids = torch.tensor(all_virus_seq_ids, dtype=torch.float)
    batch_virus_datas = torch.tensor(all_virus_data, dtype=torch.float)
    batch_virus_ids = torch.tensor(all_virus_ids, dtype=torch.float)


    return batch_virus_datas, batch_virus_ids,batch_seq_ids, batch_virus_labels
