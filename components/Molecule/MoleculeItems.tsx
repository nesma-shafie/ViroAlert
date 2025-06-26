'use client';
import { motion } from "framer-motion";
import MoleculeCard from "./MoleculeCard";
import { Drug } from '@/types';

interface MoleculeItemsProps {
    drugs: Drug[],
    title: string
}

export default function MoleculeItems({ drugs, title }: MoleculeItemsProps) {
    return (
        <>
            <motion.h1 initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="text-4xl  font-bold mb-6 text-center text-gray-800">💊 {title}</motion.h1>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
                {drugs.map((drug, index) => (
                    <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.2, duration: 0.5, ease: "easeOut" }}
                    >
                        <MoleculeCard smiles={drug.smiles} PIC50={drug.PIC50} />
                    </motion.div>
                ))}
            </div>
        </>
    )
}