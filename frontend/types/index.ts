export interface Drug {
  smiles: string;
  PIC50: number;
}
export interface ModelResult {
    confidence: number;
    explanationImages?: string[]; // optional now
    type: 'ml' | 'dl';
}