export interface Drug {
  name: string;
  smiles: string;
  kp: number;
}
export interface ModelResult {
    name: string;
    confidence: number; // 0 to 100
    explanationImages: string[]; // 2 images per model
}