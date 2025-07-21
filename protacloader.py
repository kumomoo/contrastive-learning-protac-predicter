import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

def collater(data_list):
    batch = {}
    # 收集样本标识
    name = [x["name"] for x in data_list]
    # 分别收集目标蛋白和 E3 蛋白序列
    target_seq = [x["target_seq"] for x in data_list]
    e3_seq = [x["e3_seq"] for x in data_list]
    # 收集 SMILES 数值序列，并转换为 tensor 后进行 padding
    smiles = [torch.tensor(x["smiles"]) for x in data_list]
    smiles_length = [len(x["smiles"]) for x in data_list]
    # 收集图数据列表，使用 PyG Batch 进行批处理
    graph_list = [x["graph"] for x in data_list]
    # 收集标签
    label = [x["label"] for x in data_list]

    batch["name"] = name
    batch["target_seq"] = target_seq
    batch["e3_seq"] = e3_seq
    batch["smiles"] = torch.nn.utils.rnn.pad_sequence(smiles, batch_first=True)
    batch["smiles_length"] = smiles_length
    batch["graph"] = Batch.from_data_list(graph_list)
    batch["label"] = torch.tensor(label)
    return batch


class PROTACSet(Dataset):
    def __init__(self, name_list, target_seq, e3_seq, smiles, graph, label):
        super().__init__()
        self.name = name_list
        # 将 target_seq 和 e3_seq 列表转换为字典，key 为样本名称
        self.target_seq = {name_list[i]: target_seq[i] for i in range(len(name_list))}
        self.e3_seq = {name_list[i]: e3_seq[i] for i in range(len(name_list))}
        self.smiles = smiles
        self.graph = graph
        self.label = label

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        key = self.name[idx]
        sample = {
            "name": key,
            "target_seq": self.target_seq.get(key, ""),
            "e3_seq": self.e3_seq.get(key, ""),
            "smiles": self.smiles[idx],
            "graph": self.graph[idx],
            "label": self.label[idx],
        }
        return sample
