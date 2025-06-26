import axios from "axios";

const baseURL = process.env.NEXT_PUBLIC_API_BASE_URL;


export async function getDrugs() {
  const res = await fetch(
    `test...`
  )
  if (!res.ok) {
    throw new Error('Failed to fetch Drugs')
  }
  return res.json()
}

export async function getMLResult(form: FormData, token: string) {
    const res = await axios.post(
      `${baseURL}/user/topAntiVirus`,
      form,
      {
          headers: {
              "Content-Type": "multipart/form-data",
              Authorization: `Bearer ${token}`,
          },
      }
  );
  // const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL}/user/predictHost-ML`);
  if (!res.ok) {
    throw new Error('Failed to fetch ML model result');
  }
    //         const value = response.data?.data?.top_smiles;
  return res.data?.data?.probabilty;
}

export async function getDLResult() {
  const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL}/api/models/dl`);
  if (!res.ok) {
    throw new Error('Failed to fetch DL model result');
  }
  return res.json();
}