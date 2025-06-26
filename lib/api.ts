export async function getDrugs() {
  const res = await fetch(
    `test...`
  )
  if (!res.ok) {
    throw new Error('Failed to fetch Drugs')
  }
  return res.json()
}

export async function getMLResult() {
  const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL}/api/models/ml`);
  if (!res.ok) {
    throw new Error('Failed to fetch ML model result');
  }
  return res.json();
}

export async function getDLResult() {
  const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL}/api/models/dl`);
  if (!res.ok) {
    throw new Error('Failed to fetch DL model result');
  }
  return res.json();
}