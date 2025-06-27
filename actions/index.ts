import { Drug, ModelResult } from "@/types";
import axios from "axios";
import { useRouter } from "next/navigation";

const baseURL = process.env.NEXT_PUBLIC_API_BASE_URL
const token = localStorage.getItem("token");

export async function AntiVirusGeneration(form: FormData) {
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
    localStorage.setItem("drugs", JSON.stringify(drugs));
}

export async function AntiVirusRecommendation(form: FormData) {
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
    localStorage.setItem("drugs", JSON.stringify(drugs));

}
export async function HostPrediction(form: FormData) {
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
}
export async function virusAlignment(form: FormData) {
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
}
