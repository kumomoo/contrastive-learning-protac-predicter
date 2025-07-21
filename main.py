import os
import pickle
import torch
import random
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from prepare_data import ProtacDataset
from protacloader import PROTACSet, collater
from finetune_model import ProteinEncoder, GraphEncoder, SmilesEncoder, ProtacModel
from train_and_test import train

BATCH_SIZE = 1
EPOCH = 40
TRAIN_RATE = 0.8
LEARNING_RATE = 0.001
TRAIN_NAME = "finetune"
root = "data"



SMILES_CHAR = ['[PAD]', 'C', '(', '=', 'O', ')', 'N', '[', '@', 'H', ']', '1', 'c', 'n', '/', '2', '#', 'S', 's', '+', '-', '\\', '3', '4', 'l', 'F', 'o', 'I', 'B', 'r', 'P', '5', '6', 'i', '7', '8', '9', '%', '0', 'p']

def main():
    # 加载样本标识，存放在 data/name.pkl 中
    with open(os.path.join(root, "name.pkl"), "rb") as f:
        name_list = pickle.load(f)
    
    # 加载蛋白质序列数据：目标蛋白和 E3 酶分别存放在两个文件中
    target_dataset = ProtacDataset("target_seq", root=root)
    e3_dataset = ProtacDataset("e3_seq", root=root)
    # 加载 PROTAC 的 SMILES 数值序列、图数据、标签数据
    smiles_dataset = ProtacDataset("protac_smiles", root=root)
    graph_dataset = ProtacDataset("protac_graph", root=root)
    label_dataset = ProtacDataset("label", root=root)
    
    # target_seq 与 e3_seq 为列表（顺序与 name_list 对应）
    target_seq = target_dataset.target_seq  # 目标蛋白序列列表
    e3_seq = e3_dataset.e3_seq               # E3 蛋白序列列表
    smiles_seq = smiles_dataset.smiles_seq
    graphs = graph_dataset  # 这里的 graph_dataset 支持索引访问
    labels = label_dataset.data  # 标签数据
    
    # 构造组合数据集，PROTACSet 内部会将 target_seq 和 e3_seq 列表转换为字典（key 为样本名）
    protac_set = PROTACSet(name_list, target_seq, e3_seq, smiles_seq, graphs, labels)
    data_size = len(protac_set)
    train_size = int(data_size * TRAIN_RATE)
    test_size = data_size - train_size
    
    print(f"Total data: {data_size}, Train: {train_size}, Test: {test_size}")
    
    random.seed(42)
    indices = list(range(data_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(protac_set, train_indices)
    test_dataset = Subset(protac_set, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collater, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collater, drop_last=False)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 实例化两个蛋白质编码器：一个用于 target，一个用于 E3
    target_encoder = ProteinEncoder(model_name="Rostlab/prot_t5_xl_uniref50", device=device).to(device)
    e3_encoder = ProteinEncoder(model_name="Rostlab/prot_t5_xl_uniref50", device=device).to(device)
    
    graph_encoder = GraphEncoder(num_features=1, dim=32).to(device)
    smiles_encoder = SmilesEncoder(vocab_size=len(SMILES_CHAR),
                                   max_len=50,
                                   padding_idx=SMILES_CHAR.index('[PAD]'),
                                   embedding_dim=64,
                                   dim=64,
                                   num_layers=1,
                                   bidirectional=False).to(device)

    model = ProtacModel(target_encoder, e3_encoder, graph_encoder, smiles_encoder).to(device)

    pretrained = torch.load("model/best_model.pth", map_location=device)

    graph_sd = {
        k.replace("MPNN.", "graph_encoder."): v
        for k, v in pretrained.items()
        if k.startswith("MPNN.")
    }
    model.graph_encoder.load_state_dict(graph_sd, strict=False)

    smiles_sd = {
        k.replace("SMIEnc.", "smiles_encoder."): v
        for k, v in pretrained.items()
        if k.startswith("SMIEnc.")
    }
    model.smiles_encoder.load_state_dict(smiles_sd, strict=False)

    
    writer = SummaryWriter(f"runs/{TRAIN_NAME}")
    model = train(model, 
                  train_loader=train_loader, 
                  valid_loader=test_loader, 
                  device=device, 
                  writer=writer, 
                  LOSS_NAME=TRAIN_NAME, 
                  batch_size=BATCH_SIZE, 
                  epoch=EPOCH, 
                  lr=LEARNING_RATE)
    
if __name__ == "__main__":
    Path("log").mkdir(exist_ok=True)
    Path("model").mkdir(exist_ok=True)
    main()
