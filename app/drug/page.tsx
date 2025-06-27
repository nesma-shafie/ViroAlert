'use client';
import MoleculeItems from '@/components/Molecule/MoleculeItems';
import { Drug } from '@/types';


export default function DrugPage() {
    const drugs = JSON.parse(localStorage.getItem("drugs") || "{}") as Drug[];
    console.log("Drugs in DrugPage:", drugs);
    return (
        <MoleculeItems drugs={drugs} title={"Drug SMILES"} />
    );
}
