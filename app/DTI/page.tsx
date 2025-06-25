"use client";

import { useState } from "react";
import axios from "axios";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { motion, AnimatePresence } from "framer-motion";
import { Textarea } from "@/components/ui/textarea";
import { Info, FlaskConical, TestTube2 } from "lucide-react";

export default function PredictAntivirusPage() {
  const [virus, setVirus] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [smiles, setSmiles] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [pic50, setPic50] = useState<number | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!virus && !file) {
      alert("Please enter a virus sequence or upload a FASTA file.");
      return;
    }
    if (!smiles) {
      alert("Please provide a drug SMILES.");
      return;
    }

    setIsLoading(true);
    try {
      const form = new FormData();
      if (file) form.append("file", file);
      else form.append("virus", virus);
      form.append("smiles", smiles);

      const token = localStorage.getItem("token");
      const response = await axios.post(
        "http://localhost:3000/ViroGen/app/user/predictAntiVirus",
        form,
        {
          headers: {
            "Content-Type": "multipart/form-data",
            Authorization: `Bearer ${token}`,
          },
        }
      );

      const value = response.data?.data?.pIC50;
      if (value !== undefined) {
        setPic50(value);
      } else {
        alert("No prediction value returned.");
      }
    } catch (error) {
      console.error(error);
      alert("Failed to get prediction. Please try again.");
    }
    setIsLoading(false);
  };

  const getColor = (val: number) => {
    if (val < 5) return "bg-red-500";
    if (val < 7) return "bg-amber-500";
    return "bg-green-500";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-blue-100 flex flex-col items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="w-full max-w-6xl space-y-10"
      >
        <div className="text-center space-y-4">
          <motion.h1
            className="text-5xl font-extrabold text-virogen-blue flex items-center justify-center gap-4"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <span className="text-5xl">🔬</span>
            Drug Target Interaction
          </motion.h1>
          <motion.p
            className="text-lg text-gray-600 max-w-2xl mx-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            Paste or upload a viral protein sequence and a drug SMILES to predict their interaction strength (pIC<sub>50</sub>).
          </motion.p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 bg-white/90 rounded-2xl shadow-xl">
            <CardHeader className="bg-blue-100">
              <CardTitle className="text-2xl font-bold text-virogen-blue">Input Data</CardTitle>
            </CardHeader>
            <CardContent className="p-6 space-y-6">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label className="flex items-center gap-2 text-virogen-blue-800 font-semibold">
                      🧬 Virus Sequence
                    </Label>
                    <Textarea
                      rows={5}
                      placeholder="Enter protein sequence..."
                      value={virus}
                      onChange={(e) => setVirus(e.target.value)}
                      disabled={file !== null || isLoading}
                      className="resize-none border-blue-200 focus:ring-virogen-light-blue"
                    />
                    <p className="text-xs text-gray-500">Or upload a FASTA file below.</p>
                    <Input
                      type="file"
                      accept=".fasta,.fa,.txt"
                      onChange={(e) => setFile(e.target.files?.[0] || null)}
                      disabled={virus.length > 0 || isLoading}
                      className="cursor-pointer border-blue-200"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className="flex items-center gap-2 text-virogen-blue-800 font-semibold">
                      💊   Drug SMILES
                    </Label>
                    <Input
                      type="text"
                      placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
                      value={smiles}
                      onChange={(e) => setSmiles(e.target.value)}
                      disabled={isLoading}
                      required
                      className="border-blue-200 focus:ring-virogen-light-blue"
                    />
                  </div>
                </div>

                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <div className="flex justify-center">
  <Button
    type="submit"
    disabled={isLoading}
    className="w-56 virogen-blue hover:virogen-light-blue text-white"
  >
    {isLoading ? "Predicting..." : "Predict pIC50"}
  </Button>
</div>
                </motion.div>
              </form>
            </CardContent>
          </Card>

          <Card className="rounded-2xl shadow-xl overflow-hidden bg-white/90">
            <CardHeader className="bg-blue-100">
              <CardTitle className="text-2xl font-bold text-virogen-blue">Prediction Result</CardTitle>
            </CardHeader>
            <CardContent className="p-6 space-y-6">
              <AnimatePresence>
                {pic50 !== null ? (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.4 }}
                    className="space-y-4"
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-virogen-blue-800">
                        Predicted pIC<sub>50</sub>
                      </h3>
                      <div className="group relative">
                        <Info className="h-4 w-4 text-gray-500" />
                        <span className="absolute left-6 bottom-full mb-1 w-56 text-xs bg-white text-gray-700 px-2 py-1 rounded-md shadow-md hidden group-hover:block z-10">
                          pIC<sub>50</sub> is the negative log of IC₅₀. Higher values imply stronger antiviral activity.
                        </span>
                      </div>
                    </div>
                    <div className="w-full h-4 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-700 ${getColor(pic50)}`}
                        style={{ width: `${(pic50 / 10) * 100}%` }}
                      ></div>
                    </div>
                    <div className={`text-3xl font-bold text-center ${getColor(pic50).replace("bg-", "text-")}`}>
                      {pic50.toFixed(3)}
                    </div>
                    <div className="text-sm text-center text-gray-600">
                      <span className="text-red-500 font-semibold">Weak</span> →
                      <span className="text-amber-500 font-semibold"> Moderate</span> →
                      <span className="text-green-500 font-semibold"> Strong</span>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
  initial={{ opacity: 0 }}
  animate={{ opacity: 1 }}
  transition={{ duration: 0.5 }}
  className="flex flex-col items-center justify-center h-48 text-gray-500 text-center"
>
  <TestTube2 className="h-12 w-12 text-blue-300 mb-2" />
  <p className="text-sm">Enter data and predict to see results.</p>
</motion.div>
                )}
              </AnimatePresence>
            </CardContent>
          </Card>
        </div>
      </motion.div>
    </div>
  );
}
