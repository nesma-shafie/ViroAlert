'use client';
import MoleculeItems from '@/components/Molecule/MoleculeItems';
import { useFormContext } from '@/context/FormContext';
import { Drug } from '@/types';


const drugs: Drug[] = [
    { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O', kp: 3.5 },
    { name: 'Paracetamol', smiles: 'CC(=O)NC1=CC=C(C=C1)O', kp: 2.8 },
    { name: 'Ibuprofen', smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', kp: 3.1 },
    { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', kp: 1.7 },
    { name: 'Tamiflu', smiles: 'CCOC(=O)C1C(C(CN1C)C=C(C)C)O', kp: 2.1 },
];

export default async function DrugPage() {
    // const { drugs } = useFormContext();
    // const { drugs } = await getDrugs();
    return (
        <MoleculeItems drugs={drugs} title={"Recommended Antiviruses"} />
    );
}
