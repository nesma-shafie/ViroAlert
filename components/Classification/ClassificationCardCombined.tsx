import { ModelResult } from "@/types";
import { motion } from "framer-motion";

export default function ClassificationCardCombined({ ml, dl }: { ml: ModelResult; dl: ModelResult }) {
    return (
        <motion.div
            className="bg-white rounded-2xl shadow-lg border p-6 max-w-[60%] mx-auto space-y-6"
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
        >
            {/* ML Model Section */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.5 }}
            >
                <h2 className="text-2xl font-semibold text-blue-700 mb-2">Virus2Vec ML</h2>
                <p className="text-gray-700 text-lg">
                    Likelihood of Human Adaptation:{" "}
                    <span className="font-bold text-green-600">
                        {ml.confidence.toFixed(1)}%
                    </span>
                </p>
            </motion.div>

            <hr className="border-gray-300" />

            {/* DL Model Section */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 0.5 }}
            >
                <h2 className="text-2xl font-semibold text-blue-700 mb-2">ViroGen DL</h2>
                <p className="text-gray-700 text-lg mb-4">
                    Likelihood of Human Adaptation:{" "}
                    <span className="font-bold text-green-600">
                        {dl.confidence.toFixed(1)}%
                    </span>
                </p>
                <div className="flex flex-col gap-4">
                    {dl.explanationImages?.map((src, i) => (
                        <motion.img
                            key={i}
                            src={src}
                            alt={`Explanation ${i + 1} `}
                            className="rounded-lg border w-full object-contain"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.6 + i * 0.2, duration: 0.4 }}
                        />
                    ))}
                </div>
            </motion.div>
        </motion.div>
    );
}
