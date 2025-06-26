// import { ModelResult } from "@/types";

// export default function ClassificationCard({ model }: { model: ModelResult }) {
//     return (
//         <div className="bg-white rounded-2xl shadow-lg border p-6 hover:shadow-xl transition">
//             <h2 className="text-2xl font-semibold text-blue-700 mb-2">
//                 {model.name}
//             </h2>

//             <p className="text-gray-700 text-lg mb-4">
//                 Likelihood of Human Adaptation:{" "}
//                 <span className="font-bold text-green-600">
//                     {model.confidence.toFixed(1)}%
//                 </span>
//             </p>

//             {model.type === "dl" && (
//                 <div className="flex flex-col gap-4">
//                     {model.explanationImages?.map((src, i) => (
//                         <img
//                             key={i}
//                             src={src}
//                             alt={`Explanation ${i + 1} for ${model.name}`}
//                             className="rounded-lg border w-full object-contain"
//                         />
//                     ))}
//                 </div>
//             )}
//         </div>
//     );
// }
