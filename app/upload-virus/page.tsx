"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { motion } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useState } from "react";
import axios from "axios";
import { useFormContext } from "@/context/FormContext";
import { Drug } from "@/types";
import { set } from "react-hook-form";

export default function uploadvirus() {
  const [virus, setVirus] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { drugs, setdrugs } = useFormContext();
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
        // Option 2: Use context or global state to share sequenceData
        // router.push("/alignment");
      }
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
        value = response.data?.data?.top_smiles;
      } else if (searchParams.get("flag") === "0") {
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
        value = response.data?.data?.drugs;
      }

      const formattedValue: Drug[] = value.map((val: [string, number]) => ({
        smiles: val[0],
        PIC50: val[1],
      }));
      if (formattedValue !== undefined) {
        setdrugs(formattedValue);
        router.push("/drug");
      } else {
        alert("No prediction value returned.");
      }
    } catch (error) {
      console.error(error);
      alert("Failed to get prediction. Please try again.");
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
                  <div className="flex items-center justify-center w-56 h-10">
                    <svg
                      className="animate-spin h-6 w-6 text-virogen-blue"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                      ></path>
                    </svg>
                    <span className="ml-2 text-virogen-blue font-semibold">
                      Loading...
                    </span>
                  </div>
                ) : searchParams.get("flag") === "2" ? (
                  <Button
                    type="submit"
                    disabled={isLoading}
                    className="w-56 virogen-blue hover:virogen-light-blue text-white"
                  >
                    Align{" "}
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    disabled={isLoading}
                    className="w-56 virogen-blue hover:virogen-light-blue text-white"
                  >
                    {searchParams.get("flag") === "1"
                      ? "Get Top Anti-Viruses"
                      : "Get Anti-Viruses"}
                  </Button>
                )}
              </div>
            </motion.div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
