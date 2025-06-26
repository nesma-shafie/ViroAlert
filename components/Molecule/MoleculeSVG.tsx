import * as OCL from 'openchemlib';

export default function MoleculeSVG({ smiles }: { smiles: string }) {
    let svg = '';

    try {
        const mol = OCL.Molecule.fromSmiles(smiles);
        svg = mol.toSVG(200, 200);
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
