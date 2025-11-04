# Tutorial

## Generate conformer with RDKIT ETKDG

```python
smiles = 'CCC(N)C'
mol = Chem.MolFromSmiles(smiles)
mol = AllChem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol)
block = Chem.MolToMolBlock(mol) # block is the data type for 3D viz
```

## View molecules in 3D

Using py3Dmol library

```python
viewer = py3Dmol.view(width=400, height=400)
viewer.addModel(block, 'mol')
viewer.setStyle({'stick': {}})
viewer.zoomTo()
```

Using RDKIt Draw module

```python
esomeprazole = Chem.MolFromSmiles('COc1ccc2[n-]c([S@@+]([O-])Cc3ncc(C)c(OC)c3C)nc2c1')
AllChem.EmbedMolecule(esomeprazole, AllChem.ETKDGv3())
IPythonConsole.drawMol3D(esomeprazole)
```

## The PyG Data class

- x (shape [num_atoms, 11]): node features per atom. In PyG’s QM9, this is an 11-dim feature vector (element one-hot + simple chemistry indicators).
- edge_index (shape [2, num_directed_edges]): the graph connectivity in COO format. Undirected bonds are stored twice (i→j and j→i). For methane there are 4 C–H bonds → 8 directed edges.
- edge_attr (shape [num_directed_edges, 4]): edge (bond) features. In QM9 it’s a 4-dim one-hot for bond type (single/double/triple/aromatic).
- pos (shape [num_atoms, 3]): 3D coordinates. In QM9 this is your reference geometry (so, ref_pos).
- z (shape [num_atoms]): atomic numbers (e.g., C=6, H=1).
- smiles: the SMILES string
- name: a dataset identifier (QM9 uses gdb_* names).
- idx: an integer index
