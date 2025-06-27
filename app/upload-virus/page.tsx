"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { motion } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useState } from "react";
import axios from "axios";
import { Drug, ModelResult } from "@/types";
import LoadingButton from "@/components/LoadingButton";
import FormButton from "@/components/FormButton";

export default function uploadvirus() {
  const [virus, setVirus] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();
  const baseURL = process.env.NEXT_PUBLIC_API_BASE_URL;
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!virus && !file) {
      alert("Please enter a virus sequence or upload a FASTA file.");
      return;
    }
    setIsLoading(true);
    try {
      const form = new FormData();
      if (file) form.append("file", file);
      else form.append("virus", virus);
      const token = localStorage.getItem("token");
      //flag
      let value;
      // Alignment
      if (searchParams.get("flag") === "2") {
        const response = await axios.post(`${baseURL}/user/align`, form, {
          headers: {
            "Content-Type": "multipart/form-data",
            Authorization: `Bearer ${token}`,
          },
        });
        const input = response.data?.input;
        const closest_matches = response.data?.closest_matches;
        //set input and clostest matches to send them to alignment page without local storage
        const sequenceData = {
          input: input,
          closest_matches: closest_matches,
        };
        //set sequenceData to local storage
        localStorage.setItem("sequenceData", JSON.stringify(sequenceData));
        router.push(`/alignment`);
        return;
      }
      // Anti-Virus Prediction
      if (searchParams.get("flag") === "1") {
        const response = await axios.post(
          `${baseURL}/user/topAntiVirus`,
          form,
          {
            headers: {
              "Content-Type": "multipart/form-data",
              Authorization: `Bearer ${token}`,
            },
          }
        );
        const drugs: Drug[] = response.data?.smiles.map((val: [string, number]) => ({
          smiles: val[0],
          PIC50: val[1],
        }));
        localStorage.removeItem("drugs");
        localStorage.setItem("drugs", JSON.stringify(drugs));
        router.push("/drug");
        return;
      }
      // Anti-Virus Generation
      else if (searchParams.get("flag") === "0") {
        const response = await axios.post(
          `${baseURL}/user/generateAntiVirus`,
          form,
          {
            headers: {
              "Content-Type": "multipart/form-data",
              Authorization: `Bearer ${token}`,
            },
          }
        );
        const drugs: Drug[] = response.data?.smiles.map((val: [string, number]) => ({
          smiles: val[0],
          PIC50: val[1],
        }));
        console.log("Drugs:", drugs);
        localStorage.removeItem("drugs");
        localStorage.setItem("drugs", JSON.stringify(drugs));
        router.push("/drug");
        return;
      }
      // Host Prediction
      else if (searchParams.get("flag") === "3") {
        // Deep Learning Prediction
        const dlres = await axios.post(
          `${baseURL}/user/predictHost`,
          form,
          {
            headers: {
              "Content-Type": "multipart/form-data",
              Authorization: `Bearer ${token}`,
            },
          }
        );
        const DeepLearningResult: ModelResult = {
          confidence: dlres.data?.probability,
          explanationImages: dlres.data?.img,
          type: "dl",
        };
        localStorage.setItem("DeepLearningResult", JSON.stringify(DeepLearningResult));
        // Machine Learning Prediction
        const mlres = await axios.post(
          `${baseURL}/user/predictHost-ML`,
          form,
          {
            headers: {
              "Content-Type": "multipart/form-data",
              Authorization: `Bearer ${token}`,
            },
          }
        );
        const MachineLearningResult: ModelResult = {
          confidence: mlres.data?.probability,
          type: "ml",
        };

        localStorage.setItem("MachineLearningResult", JSON.stringify(MachineLearningResult));
        router.push(`/classification`);
        return;
      }

    } catch (error) {
      console.error(error);
      if (typeof error === "object" && error !== null && "response" in error) {
        // @ts-ignore
        alert(`Failed to get prediction. ${error.response?.data?.message || error.message}.`);
      } else {
        alert("Failed to get prediction. An unknown error occurred.");
      }
    }
    setIsLoading(false);
  };
  return (
    <div className="max-w-[50%] mx-auto p-6">
      <Card className="bg-white/90 rounded-2xl shadow-xl">
        <CardHeader className="bg-blue-100">
          <CardTitle className="text-2xl font-bold text-virogen-blue">
            Input Data
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 space-y-6">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2 w-full">
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
              <p className="text-xs text-gray-500">
                Or upload a FASTA file below.
              </p>
              <Input
                type="file"
                accept=".fasta,.fa,.txt"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                disabled={virus.length > 0 || isLoading}
                className="cursor-pointer border-blue-200"
              />
            </div>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <div className="flex justify-center">
                {isLoading ? (
                  <LoadingButton />
                ) : (
                  <FormButton
                    isLoading={isLoading}
                    text={
                      searchParams.get("flag") === "3"
                        ? "Predict Host"
                        : searchParams.get("flag") === "2"
                          ? "Align"
                          : searchParams.get("flag") === "1"
                            ? "Get Top Anti-Viruses"
                            : "Get Anti-Viruses"
                    }
                  />
                )}
              </div>
            </motion.div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
