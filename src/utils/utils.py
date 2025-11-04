from torch_geometric.data import Data
from tqdm import tqdm
from typing import List
import torch
from rdkit.Chem import AllChem
from rdkit import Chem
from torch_geometric.utils import one_hot
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from typing import List, Union
import torch
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np

class DataTransform:
    def __init__(self):
        self.ATOM_MAP = {
                "H": 0,
                "C": 1,
                "N": 2,
                "O": 3,
                "F": 4,
                "P": 5,
                "S": 6,
                "Cl": 7,
                "Br": 8,
                "I": 9,
            }

        self.BOND_MAP = {
            'SINGLE': 0,
            'DOUBLE': 1, 
            'TRIPLE': 2,  
            'AROMATIC':3 
        }
        self.ATOM_ONEHOT = np.eye(len(self.ATOM_MAP), dtype=np.float32)
        self.BOND_ONEHOT = np.eye(len(self.BOND_MAP), dtype=np.float32)

    def smiles_to_graph(self, smiles: Union[List[str], str]):
        r"""Convert one or more SMILES strings into PyTorch Geometric ``Data`` graphs.

        Each molecule is standardized, hydrogenated, 3D-embedded (ETKDGv3), and
        MMFF94-optimized. Node features are atom-type one-hot vectors derived from
        ``self.ATOM_ONEHOT``. Edge features are bond-type one-hot vectors from
        ``self.BOND_ONEHOT``. Edges are stored in COO format with both directions.
        3D coordinates are stored in ``pos`` as float32.

        Args:
            smiles (Union[List[str], str]): A single SMILES string or a list of SMILES.

        Returns:
            List[torch_geometric.data.Data]: One ``Data`` object per successfully
            processed molecule. The list may be shorter than the input if parsing,
            embedding, or optimization fails.

        Each ``Data`` contains:
            - **x** (*Tensor*: ``[N, F_node]``) — Atom one-hot features.
            - **edge_index** (*LongTensor*: ``[2, E]``) — Directed edges (both directions).
            - **edge_attr** (*Tensor*: ``[E, F_edge]``) — Bond one-hot features.
            - **pos** (*Tensor*: ``[N, 3]``) — 3D coordinates (Å), float32.
            - **smi** (*str*) — Original SMILES string.
            - **labels** (*Dict[int, str]*) — Map from node index to atomic symbol.
            - **idx** (*int*) — Input index.

        Notes:
            - Standardization uses ``rdMolStandardize.StandardizeSmiles``.
            - Hydrogens are added before embedding.
            - 3D embedding uses ``AllChem.ETKDGv3()`` and geometry is refined with
            ``AllChem.MMFFOptimizeMolecule``.
            - Failures are caught and skipped; no exception is raised.
            - ETKDG is stochastic unless you set RDKit random seeds.

        Shape:
            - ``N``: number of atoms.
            - ``E``: number of directed edges (twice the number of bonds).
            - ``F_node``: length of ``self.ATOM_ONEHOT`` feature vector.
            - ``F_edge``: length of ``self.BOND_ONEHOT`` feature vector.

        Example:
            >>> graphs = model.smiles_to_graph(["CCO", "c1ccccc1"])
            >>> g = graphs[0]
            >>> g.x.shape, g.edge_index.shape, g.edge_attr.shape, g.pos.shape
            (torch.Size([N, F_node]), torch.Size([2, E]), torch.Size([E, F_edge]), torch.Size([N, 3]))
            """
        if isinstance(smiles, str):
                smiles = [smiles]

        graphs = []
        for idx, smi in tqdm(enumerate(smiles), desc='Converting SMILES into Graphs', total=len(smiles)):
            try:
                std_smi = rdMolStandardize.StandardizeSmiles(smi)
                mol = Chem.MolFromSmiles(std_smi)
                if mol is None:
                    continue
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) # type: ignore
                AllChem.MMFFOptimizeMolecule(mol) # type: ignore
                conf = mol.GetConformer()
                pos_matrix = np.asarray(conf.GetPositions(), dtype=np.float32)  # (N,3)
            except Exception:
                continue

            edge_index = []
            bond_types = []

            # node features
            atom_idx_mapping = [self.ATOM_MAP[a.GetSymbol()] for a in mol.GetAtoms()]
            x = torch.tensor(self.ATOM_ONEHOT[atom_idx_mapping], dtype=torch.float32)
            labels = {i: atom.GetSymbol() for i, atom in enumerate(mol.GetAtoms())}  #  useful for plotting later maps idx to atom symbol
            # edges and edge features (both directions => COO format )
            for b in mol.GetBonds():
                i = b.GetBeginAtomIdx()
                j = b.GetEndAtomIdx()
                bt = self.BOND_MAP[str(b.GetBondType())]
                edge_index.append((i, j))
                edge_index.append((j, i))
                bond_types.append(bt)
                bond_types.append(bt)

            if edge_index:
                ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
                edge_attr = torch.tensor(self.BOND_ONEHOT[bond_types], dtype=torch.float32)  # [E, F]
            else:
                ei = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(self.BOND_MAP)), dtype=torch.float32)

            data = Data(
                x=x,
                edge_index=ei,
                edge_attr=edge_attr,
                pos=torch.tensor(pos_matrix, dtype=torch.float32),  # shape [N,3]
                smi=smi,
                labels = labels,
                idx=idx,
            )
            graphs.append(data)

        return graphs