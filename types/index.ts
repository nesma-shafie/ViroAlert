export interface Drug {
  name: string;
  smiles: string;
  kp: number;
}
export interface ModelResult {
    confidence: number;
    explanationImages?: string[]; // optional now
    type: 'ml' | 'dl';
}