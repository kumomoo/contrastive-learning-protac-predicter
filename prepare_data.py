import torch
import pickle
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import torch.nn.functional as F
from rdkit import Chem
from pathlib import Path

# 定义 SMILES 字符表
SMILES_CHAR = ['[PAD]', 'C', '(', '=', 'O', ')', 'N', '[', '@', 'H', ']', '1', 'c', 'n', '/', '2', '#', 'S', 's', '+', '-', '\\', '3', '4', 'l', 'F', 'o', 'I', 'B', 'r', 'P', '5', '6', 'i', '7', '8', '9', '%', '0', 'p']

# 定义 bond type 到 one-hot 索引的映射
BOND_TYPE_TO_IDX = {
    Chem.rdchem.BondType.SINGLE:   0,
    Chem.rdchem.BondType.DOUBLE:   1,
    Chem.rdchem.BondType.TRIPLE:   2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
NUM_BOND_TYPES = 5  # 4 种已知键 + 1 种“其它”

def trans_smiles(smi):
    """
    将 SMILES 字符串转换为字符索引列表，
    对每个字符查找其在 SMILES_CHAR 中的索引，
    若不存在则用 len(SMILES_CHAR) 作为默认值
    """
    temp = list(smi)
    temp = [SMILES_CHAR.index(ch) if ch in SMILES_CHAR else len(SMILES_CHAR) for ch in temp]
    return temp

def smiles_to_graph(smile: str) -> Data:
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"无法解析 SMILES: {smile!r}")
    except ValueError:
        # 只有 SMILES 完全没法解析时才回退
        return Data(
            x=torch.zeros((1, 1), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, NUM_BOND_TYPES), dtype=torch.float)
        )

    atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    edge_attr  = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]

        idx = BOND_TYPE_TO_IDX.get(bond.GetBondType(), NUM_BOND_TYPES - 1)
        one_hot = F.one_hot(torch.tensor(idx), num_classes=NUM_BOND_TYPES).float()
        edge_attr += [one_hot, one_hot]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2,0), dtype=torch.long)
    edge_attr  = torch.stack(edge_attr, dim=0)               if edge_attr  else torch.empty((0, NUM_BOND_TYPES), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class ProtacDataset(InMemoryDataset):
    """
    数据集包含以下几部分：
      1. 蛋白质序列数据：来自 "seq_train.csv" 中的 sequence_target 与 sequence_E3 列，
         分别保存在 "target_seq.pkl"（target序列）和 "e3_seq.pkl"（E3序列）中。
      2. PROTAC 分子的 SMILES 数值序列：来自 "protac.csv" 中的 Smiles 列，经过 trans_smiles 转换
      3. PROTAC 分子的图数据：同样来自 "protac.csv" 的 Smiles 列，经过 smiles_to_graph 转换
      4. 标签数据：来自 "protac.csv" 中的 Degradation 列（假设 "Good" 转为 1，"Bad" 转为 0）
    """
    def __init__(self, name, root="data"):
        super().__init__(root)
        if name == "target_seq":
            with open(self.processed_paths[0], "rb") as f:
                self.target_seq = pickle.load(f)
        elif name == "e3_seq":
            with open(self.processed_paths[1], "rb") as f:
                self.e3_seq = pickle.load(f)
        elif name == "protac_smiles":
            with open(self.processed_paths[2], "rb") as f:
                self.smiles_seq = pickle.load(f)
        elif name == "protac_graph":
            self.data, self.slices = torch.load(self.processed_paths[3])
        elif name == "label":
            self.data = torch.load(self.processed_paths[4])
    
    @property
    def processed_file_names(self):
        return [
            "target_seq.pkl",    # 目标蛋白序列（target）
            "e3_seq.pkl",         # E3 蛋白序列
            "protac_smiles.pkl",  # PROTAC 分子的 SMILES 数值序列
            "protac_graph.pt",    # PROTAC 分子的图数据
            "label.pt",           # 标签数据
        ]
    
    def process(self):
        # 1. 处理蛋白质序列数据：读取 seq_train.csv
        seq_df = pd.read_csv("data/seq_train.csv")
        target_seq = []
        e3_seq = []
        for i, row in seq_df.iterrows():
            # 假设每一行都有 "sequence_target" 和 "sequence_E3"
            target_seq.append(row["sequence_target"])
            e3_seq.append(row["sequence_E3"])
        with open(self.processed_paths[0], "wb") as f:
            pickle.dump(target_seq, f)
        with open(self.processed_paths[1], "wb") as f:
            pickle.dump(e3_seq, f)
        
        # 2. 处理 PROTAC 的 SMILES 数据及图数据：读取 protac.csv
        protac_df = pd.read_csv("data/protac.csv")
        smiles_seq = []
        protac_graphs = []
        labels = []
        for i, row in protac_df.iterrows():
            smi = row["Smiles"]
            lab = row["Degradation"]
            # 标签转换：假设 "Good" 对应 1，其它视为 0
            labels.append(lab)
            # 标准化 SMILES
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                smi_canonical = ""
            else:
                smi_canonical = Chem.MolToSmiles(mol)
            # 转换为字符索引序列
            smiles_seq.append(trans_smiles(smi_canonical.strip()))
            # 转换为图数据
            try:
                graph = smiles_to_graph(smi_canonical.strip())
            except Exception as e:
                graph = Data(x=torch.zeros((1, 1)),
                             edge_index=torch.empty((2, 0), dtype=torch.long),
                             edge_attr=torch.empty((0, 1)))
            protac_graphs.append(graph)
        
        # 保存 SMILES 数值序列
        with open(self.processed_paths[2], "wb") as f:
            pickle.dump(smiles_seq, f)
        # 保存图数据
        data, slices = self.collate(protac_graphs)
        torch.save((data, slices), self.processed_paths[3])
        # 保存标签
        torch.save(labels, self.processed_paths[4])
    
if __name__ == "__main__":
    target_dataset = ProtacDataset("target_seq")
    e3_dataset = ProtacDataset("e3_seq")
    smiles_dataset = ProtacDataset("protac_smiles")
    graph_dataset = ProtacDataset("protac_graph")
    label_data = ProtacDataset("label")
    
    print("蛋白质序列样例：", target_dataset.target_seq[:2])
    print("E3 蛋白序列样例：", e3_dataset.e3_seq[:2])
    print("SMILES 数值序列样例：", smiles_dataset.smiles_seq[:2])
    print("图数据：", graph_dataset.data)
    print("标签：", label_data.data)
