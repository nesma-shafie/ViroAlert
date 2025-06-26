'use client';

import MoleculeSVG from "./MoleculeSVG";

export default function MoleculeCard({ smiles, PIC50 }: { smiles: string; PIC50: number }) {
    return (
        <div className="bg-white p-4 rounded-2xl shadow-lg border hover:shadow-xl transition">
            {/* <h2 className="text-xl font-semibold mb-2 text-blue-800">{name}</h2> */}
            <MoleculeSVG smiles={smiles} />
            <p className="mt-2 text-sm text-gray-600 break-all">
                <strong>SMILES :</strong> {smiles}
            </p>
            <p className="mt-2 text-sm text-gray-600">
                <strong>PIC50 :</strong> {PIC50.toFixed(1)} 
            </p>
        </div>
    );
}