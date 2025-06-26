import MoleculeCard from "./MoleculeCard";
import { Drug } from '@/types';

interface MoleculeItemsProps {
    drugs: Drug[],
    title: string
}

export default function MoleculeItems({ drugs, title }: MoleculeItemsProps) {
    return (
        <>
            <h1 className="text-4xl  font-bold mb-6 text-center text-gray-800">💊 {title}</h1>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
                {drugs.map((drug, index) => (
                    <MoleculeCard key={index} name={drug.name} smiles={drug.smiles} kp={drug.kp} />
                ))}
            </div>
        </>
    )
}