'use client';
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { motion } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useState } from "react";
import axios from "axios";
import { useFormContext } from "@/context/FormContext";


export default function uploadvirus() {
    const [virus, setVirus] = useState("");
    const [file, setFile] = useState<File | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const { setdrugs } = useFormContext();
    const router = useRouter();
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
            const response = await axios.post(
                `${baseURL}/user/predictAntiVirus`,
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
                setdrugs(response.data.data.drugs);
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
        <div className='max-w-[50%] mx-auto p-6'>
            <Card className="bg-white/90 rounded-2xl shadow-xl">
                <CardHeader className="bg-blue-100">
                    <CardTitle className="text-2xl font-bold text-virogen-blue">Input Data</CardTitle>
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
                            <p className="text-xs text-gray-500">Or upload a FASTA file below.</p>
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
                                <Button
                                    type="submit"
                                    disabled={isLoading}
                                    className="w-56 virogen-blue hover:virogen-light-blue text-white"
                                >
                                    {isLoading ? "Predicting..." : "Get Anti-Virus"}
                                </Button>
                            </div>
                        </motion.div>
                    </form>
                </CardContent>
            </Card>
        </div>
    )
}

