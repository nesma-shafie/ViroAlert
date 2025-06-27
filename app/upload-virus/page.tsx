"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { motion } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useState } from "react";
import LoadingButton from "@/components/LoadingButton";
import FormButton from "@/components/FormButton";
import { AntiVirusGeneration, AntiVirusRecommendation, HostPrediction, virusAlignment } from "@/actions";

export default function uploadvirus() {
  const [virus, setVirus] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();
  const baseURL = process.env.NEXT_PUBLIC_API_BASE_URL;
  async function VirusUpload(flag: string, virus: string, file: File | null) {
    if (!virus && !file) {
      alert("Please enter a virus sequence or upload a FASTA file.");
      return;
    }
    const form = new FormData();
    if (file) form.append("file", file);
    else form.append("virus", virus);
    // Alignment
    if (flag === "2") {
      await virusAlignment(form);
      router.push(`/alignment`);
      return;
    }
    // Anti-Virus Prediction
    if (flag === "1") {
      await AntiVirusRecommendation(form);
      router.push(`/drug`);
      return;
    }
    // Anti-Virus Generation
    else if (flag === "0") {
      await AntiVirusGeneration(form);
      router.push(`/drug`);
      return;
    }
    // Host Prediction
    else if (flag === "3") {
      await HostPrediction(form);
      router.push(`/classification`);
      return;
    }
  }
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      await VirusUpload(searchParams.get("flag") as string, virus, file);
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
