from tqdm import tqdm
import numpy as np 
from collections import Counter
from collections import defaultdict


def remove_duplicateSeq(datas, ids):
    unique_dict = {}  
    for data, id_ in zip(datas, ids):
        unique_dict[data] = id_  
    return list(unique_dict.keys()),list(unique_dict.values())

def remove_duplicateIds(datas,ids):
    datas = np.array(datas)
    ids = np.array(ids)
    # Find duplicated ids 
    unique_ids, counts = np.unique(ids, return_counts=True)
    duplicate_ids = {id for id in unique_ids[counts > 1]}  
    
    
    indices_to_remove = np.array([
        ( id_val in duplicate_ids and 'X' in data_val) 
        for id_val, data_val in tqdm(zip(ids, datas))
    ])
    
    filtered_data = datas[~indices_to_remove]
    filtered_id = ids[~indices_to_remove]

     # Find unique id pairs and keep only the first occurrence
    _,unique_indices = np.unique(filtered_id, return_index=True)
    
    filtered_data = filtered_data[unique_indices]
    filtered_id = filtered_id[unique_indices]

    return filtered_data,filtered_id

def remove_x_seq( datas, ids):
    # Step 1: Compute X frequencies for all sequences
    Xfreqs = np.array([
        Counter(seq).get("X", 0) / len(seq) if len(seq) != 0 else 100
        for seq in datas
    ])
    # Step 2: Create a mask to filter out sequences where X frequency > 0.1
    mask = Xfreqs > 0.1  
    return (
        np.array(datas)[mask], 
        np.array(ids)[mask]
    )
    
def remove_incomplete_seq(datas, ids):
    filtered_datas=[]
    filtered_ids=[]
    for data, id_ in zip(datas, ids):
        if len(data) > 200:
            filtered_datas.append(data)
            filtered_ids.append(id_)

    return filtered_datas, filtered_ids
    
def hamming_distance_np(arr1, arr2):
    return np.sum(arr1 != arr2, axis=1)

def process_length_group(sequences, ids, threshold):
    if not sequences:
        return [], []
    
    # Sort sequences to prioritize those with "human" in their ID
    sorted_data = sorted(zip(sequences, ids), key=lambda x: "Human"not in x[1])  
    sorted_sequences, sorted_ids = zip(*sorted_data)  

    unique_sequences = []
    unique_ids = []

    np_sequences = np.array([list(seq) for seq in sorted_sequences], dtype="<U1")  

    for i in tqdm(range(len(np_sequences))):
        seq = np_sequences[i]
        id_ = sorted_ids[i]

        if not unique_sequences:
            unique_sequences.append(seq)
            unique_ids.append(id_)
            continue

        np_unique = np.array(unique_sequences, dtype="<U1")

        dists = hamming_distance_np(np_unique, seq)
        normalized_dists = dists / len(seq)

        if np.all(normalized_dists >= threshold):
            unique_sequences.append(seq)
            unique_ids.append(id_)

           

    # Convert back to strings
    unique_sequences = ["".join(seq) for seq in unique_sequences]
    
    return unique_sequences, unique_ids

def remove_similar_sequences(sequences, ids, threshold=0.01/2):
    if len(sequences) == 0:
        return [], []

    # Group sequences by length
    length_groups = defaultdict(list)
    for seq, id_ in zip(sequences, ids):
        length_groups[len(seq)].append((seq, id_))

    # Process each group separately
    unique_sequences = []
    unique_ids = []

    for length, seq_group in length_groups.items():
        group_sequences, group_ids = zip(*seq_group)
        u_seqs, u_ids = process_length_group(group_sequences, group_ids, threshold)
        unique_sequences.extend(u_seqs)
        unique_ids.extend(u_ids)

    return unique_sequences, unique_ids
