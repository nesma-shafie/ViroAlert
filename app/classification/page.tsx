
import ClassificationItems from "@/components/Classification/ClassificationItems";
import { ModelResult } from "@/types"

const models: ModelResult[] = [
    {
        name: 'Virus2Vec (ML)',
        confidence: 78.5,
        explanationImages: ['/bag-level.png', '/instance-level.png'],
    },
    {
        name: 'ViroGen (DL)',
        confidence: 91.2,
        explanationImages: ['/bag-level.png', '/instance-level.png'],
    },
];

export default async function ClassificationPage() {
    // const [ml, dl] = await Promise.all([getMLResult(), getDLResult()]);
    return (
        <div>
            <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">
                🧬 Virus Human Adaptation Prediction
            </h1>
            <ClassificationItems models={models} />
        </div>
    );
}
