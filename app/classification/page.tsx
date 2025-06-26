'use client';
import ClassificationCardCombined from "@/components/Classification/ClassificationCardCombined";
import { ModelResult } from "@/types"
import { motion } from "framer-motion";

const models: ModelResult[] = [
    {
        confidence: 78.5,
        type: 'ml',
    },
    {
        confidence: 91.2,
        explanationImages: ['/bag-level.png', '/instance-level.png'],
        type: 'dl',
    },
];

export default async function ClassificationPage() {
    // const [ml, dl] = await Promise.all([getMLResult(), getDLResult()]);
    return (
        <div>
            <motion.h1
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="text-4xl font-bold text-center mb-8 text-gray-800"
            >
                🧬 Virus Human Adaptation Prediction
            </motion.h1>
            <ClassificationCardCombined ml={models[0]} dl={models[1]} />
        </div>
    );
}
