import * as OCL from 'openchemlib';

export default function MoleculeSVG({ smiles }: { smiles: string }) {
    let svg = '';

    try {
        // Parse the molecule from SMILES
        let mol = OCL.Molecule.fromSmiles(smiles);

        // Clean the molecule structure and remove unknown chirality
        mol.ensureHelperArrays(OCL.Molecule.cHelperNeighbours);
        mol.setFragment(false); // Treat as full molecule, not a fragment

        // Optionally sanitize chirality by removing stereo info
        for (let i = 0; i < mol.getAllAtoms(); i++) {
            mol.setAtomConfigurationUnknown(i, false); // Disable ? marks
            mol.setAtomParity(i, 0, false); // Clear stereo parity
        }

        // Optionally remove unknown bond types (usually caused by question marks)
        for (let i = 0; i < mol.getAllBonds(); i++) {
            if (mol.getBondType(i) === OCL.Molecule.cBondTypeDown) {
                mol.setBondType(i, OCL.Molecule.cBondTypeSingle);
            }
        }

        svg = mol.toSVG(200, 200, 'element'); // You can change to 'element' or '' as needed
        // Remove "unknown chirality" text element from the SVG string
        svg = svg.replace(/<text[^>]*>.*?unknown chirality.*?<\/text>/gi, '');
    } catch (e) {
        svg = '<text x="10" y="20" fill="red">Invalid SMILES</text>';
    }

    return (
        <div
            className="w-full flex justify-center"
            dangerouslySetInnerHTML={{ __html: svg }}
        />
    );
}
