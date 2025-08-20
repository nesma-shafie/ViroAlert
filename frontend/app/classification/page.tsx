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
    //     const handleSubmit = async (e: React.FormEvent) => {
    //     e.preventDefault();
    //     if (!virus && !file) {
    //         alert("Please enter a virus sequence or upload a FASTA file.");
    //         return;
    //     }
    //     setIsLoading(true);
    //     try {
    //         const form = new FormData();
    //         if (file) form.append("file", file);
    //         else form.append("virus", virus);
    //         const token = localStorage.getItem("token");
    //         const response = await axios.post(
    //             `${baseURL}/user/topAntiVirus`,
    //             form,
    //             {
    //                 headers: {
    //                     "Content-Type": "multipart/form-data",
    //                     Authorization: `Bearer ${token}`,
    //                 },
    //             }
    //         );
    //         const value = response.data?.data?.top_smiles;
    //         const formattedValue: Drug[] = value.map((val: [string, number]) => ({
    //             smiles: val[0],
    //             PIC50: val[1],
    //         }));
    //         if (formattedValue !== undefined) {
    //             setdrugs(formattedValue);
    //             router.push("/drug");
    //         } else {
    //             alert("No prediction value returned.");
    //         }
    //     } catch (error) {
    //         console.error(error);
    //         alert("Failed to get prediction. Please try again.");
    //     }
    //     setIsLoading(false);
    // };
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
