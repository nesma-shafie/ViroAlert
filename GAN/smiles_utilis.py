from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SanitizeMol
from rdkit import DataStructs

# Validity metric using RDKit
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
    
def get_tanimoto_similarity(smiles1,smiles2):
    # Convert to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    # Generate Morgan fingerprints (ECFP)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
    # Calculate Tanimoto similarity
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity