'use client';
import ClassificationCardCombined from "@/components/Classification/ClassificationCardCombined";
import { ModelResult } from "@/types"
import { motion } from "framer-motion";

export default function ClassificationPage() {
    const dlmodel = JSON.parse(localStorage.getItem("DeepLearningResult") || "{}") as ModelResult;
    const mlmodel = JSON.parse(localStorage.getItem("MachineLearningResult") || "{}") as ModelResult;
    return (
        <div>
            <motion.h1
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="text-4xl font-bold text-center mb-8 bg-gradient-to-r from-cyan-600 to-blue-600 text-transparent bg-clip-text"
            >
                🧬 Virus Human Adaptation Prediction
            </motion.h1>
            <ClassificationCardCombined ml={mlmodel} dl={dlmodel} />
        </div>
    );
}
