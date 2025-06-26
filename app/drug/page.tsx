'use client';
import MoleculeItems from '@/components/Molecule/MoleculeItems';
import { useFormContext } from '@/context/FormContext';
import router from 'next/navigation';
import { root } from 'postcss';
import { useRouter } from "next/navigation";

// import { Drug } from '@/types';


// const drugs: Drug[] = [
//     { name: 'Aspirin', smiles: 'CCc1cn([C@@H]2O[C@H](CNC(=O)C3c4ccccc4Oc4c(Cl)cccc43)[C@@H](O)[C@@H]2F)c(=O)[nH]c1=O', kp: 3.5 },
//     { name: 'Paracetamol', smiles: 'CC(=O)NC1=CC=C(C=C1)O', kp: 2.8 },
//     { name: 'Ibuprofen', smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', kp: 3.1 },
//     { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', kp: 1.7 },
//     { name: 'phentytoin', smiles: 'C1=CC=C(C=C1)C2(C(=O)NC(=O)N2)C3=CC=CC=C3', kp: 2.1 },
// ];

export default function DrugPage() {
    const { drugs } = useFormContext();
    const router = useRouter();
    if (!drugs || drugs.length === 0) {
        router.push('/upload-virus'); // Redirect to home if no drugs are available
    }
    console.log("Drugs in DrugPage:", drugs);
    return (
        <MoleculeItems drugs={drugs} title={"Drug SMILES"} />
    );
}
