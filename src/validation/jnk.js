import * as RDKit from 'rdkit';
import html2canvas from 'html2canvas';

async function generateAndSaveSmilesImage(smiles, outputFileName = 'smiles_image.png') {
    try {
        // Initialize RDKit
        await RDKit.init();

        // Create a molecule from SMILES string
        const mol = RDKit.Molecule.fromSmiles(smiles);
        if (!mol) {
            throw new Error('Invalid SMILES string');
        }

        // Create a canvas element
        const canvas = document.createElement('canvas');
        canvas.width = 300;
        canvas.height = 300;
        document.body.appendChild(canvas);

        // Generate 2D coordinates
        mol.compute2DCoords();

        // Draw molecule to canvas
        const drawOptions = {
            width: 300,
            height: 300,
            bondLineWidth: 1,
            backgroundColor: [1, 1, 1, 1] // White background
        };
        mol.drawToCanvas(canvas, drawOptions);

        // Convert canvas to image and trigger download
        const dataUrl = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = outputFileName;
        link.click();

        // Clean up
        document.body.removeChild(canvas);
        mol.delete();

        console.log(`Image saved as ${outputFileName}`);
    } catch (error) {
        console.error('Error generating SMILES image:', error);
    }
}

// Example usage
generateAndSaveSmilesImage('c1ccccc1', 'benzene.png');