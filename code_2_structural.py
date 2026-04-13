"""
Code 2: Exactly 40 structural limits rigorously enforced.
"""
from typing import Dict
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Fragments

def get_structural_features(smiles: str) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {f"struct_{i}": 0.0 for i in range(40)}

    feats = {}
    
    # 1-19: Fragments
    feats["fr_benzene"] = float(Fragments.fr_benzene(mol))
    feats["fr_ether"] = float(Fragments.fr_ether(mol))
    feats["fr_amide"] = float(Fragments.fr_amide(mol))
    feats["fr_NH0"] = float(Fragments.fr_NH0(mol))
    feats["fr_NH1"] = float(Fragments.fr_NH1(mol))
    feats["fr_NH2"] = float(Fragments.fr_NH2(mol))
    feats["fr_COO"] = float(Fragments.fr_COO(mol))
    feats["fr_C_O"] = float(Fragments.fr_C_O(mol))
    feats["fr_COO2"] = float(Fragments.fr_COO2(mol))
    feats["fr_phenol"] = float(Fragments.fr_phenol(mol))
    feats["fr_pyridine"] = float(Fragments.fr_pyridine(mol))
    feats["fr_piperdine"] = float(Fragments.fr_piperdine(mol))
    feats["fr_piperzine"] = float(Fragments.fr_piperzine(mol))
    feats["fr_Ar_N"] = float(Fragments.fr_Ar_N(mol))
    feats["fr_Ar_OH"] = float(Fragments.fr_Ar_OH(mol))
    feats["fr_halogen"] = float(Fragments.fr_halogen(mol))
    feats["fr_alkyl_halide"] = float(Fragments.fr_alkyl_halide(mol))
    feats["fr_aniline"] = float(Fragments.fr_aniline(mol))
    feats["fr_ester"] = float(Fragments.fr_ester(mol))
    
    # 20-33: Rings & Bounds
    feats["NumRotatableBonds"] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    feats["NumRings"] = float(rdMolDescriptors.CalcNumRings(mol))
    feats["NumAromaticRings"] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    feats["NumAliphaticRings"] = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
    feats["NumSaturatedRings"] = float(rdMolDescriptors.CalcNumSaturatedRings(mol))
    feats["NumHeterocycles"] = float(rdMolDescriptors.CalcNumHeterocycles(mol))
    feats["NumAromaticHeterocycles"] = float(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
    feats["NumAliphaticHeterocycles"] = float(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
    feats["NumSaturatedHeterocycles"] = float(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
    feats["fr_ketone"] = float(Fragments.fr_ketone(mol))
    feats["fr_lactone"] = float(Fragments.fr_lactone(mol))
    feats["fr_urea"] = float(Fragments.fr_urea(mol))
    feats["fr_aldehyde"] = float(Fragments.fr_aldehyde(mol))
    feats["NumHeavyAtoms"] = float(mol.GetNumHeavyAtoms())
    
    # 34-40: Special Stereocenters & Acceptors
    feats["NumAmideBonds"] = float(rdMolDescriptors.CalcNumAmideBonds(mol))
    feats["NumSpiroAtoms"] = float(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    feats["NumBridgeheadAtoms"] = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    feats["NumAtomStereoCenters"] = float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
    feats["NumHBA"] = float(rdMolDescriptors.CalcNumHBA(mol))
    feats["NumHBD"] = float(rdMolDescriptors.CalcNumHBD(mol))
    feats["fr_bicyclic"] = float(Fragments.fr_bicyclic(mol))

    assert len(feats) == 40, f"Error: Exact structure elements {len(feats)} not 40"
    return feats
