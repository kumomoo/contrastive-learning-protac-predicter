import torch
import torch.nn as nn
import torch.nn.functional as F
import types
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace(
        disable=lambda *args, **kwargs: (lambda fn: fn)
    )
from torch_geometric.nn import NNConv, Set2Set
from transformers import T5Tokenizer, T5EncoderModel
import re
import esm  # 需要安装 fair-esm（pip install fair-esm）

class GraphEncoder(nn.Module):
    def __init__(self, num_features, dim):
        super(GraphEncoder, self).__init__()
        self.lin0 = nn.Linear(num_features, dim)
        
        mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, dim * dim)
        )
        self.conv = NNConv(dim, dim, mlp, aggr='mean', root_weight=False)
        self.gru = nn.GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
    
    def forward(self, data):
        # data 应该包含 data.x, data.edge_index, data.edge_attr, data.batch
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        return self.set2set(out, data.batch)


class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size = 41, max_len = 50, padding_idx = 0, embedding_dim=64, dim=128, num_layers=1, bidirectional=False):
        """
        :param vocab_size: 字符表大小
        :param max_len: 序列固定长度
        :param padding_idx: 填充字符的索引
        :param embedding_dim: 嵌入向量维度
        :param dim: LSTM 隐藏层维度
        :param num_layers: LSTM 层数
        :param bidirectional: 是否双向 LSTM（预训练时需保持一致）
        """
        super(SmilesEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.dim = dim
        self.num_layers = num_layers
        
        self.encoder = nn.Sequential(
            nn.Embedding(num_embeddings=self.vocab_size,
                         embedding_dim=self.embedding_dim,
                         padding_idx=self.padding_idx),
            nn.LSTM(self.embedding_dim,
                    self.dim,
                    self.num_layers,
                    batch_first=True,
                    bidirectional=self.bidirectional)
        )
    
    def forward(self, smiles):
        # smiles: Tensor of shape (batch_size, max_len)
        feat, (_, _) = self.encoder(smiles)
        return feat[:, -1]

# class ProteinEncoder(nn.Module):
#     def __init__(self, model_name="esm2_t6_8M_UR50D", device="cuda"):
#         """
#         用于编码蛋白质序列的网络，
#         将目标蛋白或 E3 酶的序列输入后输出固定维度的 embedding 向量。

#         参数：
#           - model_name: 使用的 ESM-2 模型名称（默认使用 esm2_t6_8M_UR50D）
#           - device: 设备选择 ("cuda" 或 "cpu")
#         """
#         super(ProteinEncoder, self).__init__()
#         self.device = device
#         # 加载预训练的 ESM-2 模型和对应的字母表（alphabet）
#         self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
#         self.batch_converter = self.alphabet.get_batch_converter()
#         self.model.to(self.device)
#         self.model.eval()  # 设为评估模式，关闭 dropout 等

#     def forward(self, sequences):
#         """
#         :param sequences: 一个列表，每个元素是一个 (protein_name, protein_sequence) 元组，
#                         例如：[("target_protein", "MSEQNNTEMTFQIQRI..."), ("E3_enzyme", "MGSSHHHHHHSSGLVPRGSH...")]
#         :return: embeddings, Tensor，形状为 (num_sequences, embedding_dim)
#         """
#         # 利用 ESM 提供的 batch_converter 将序列转换为模型输入格式
#         batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
#         batch_tokens = batch_tokens.to(self.device)

#         with torch.no_grad():
#             # 使用模型的最后一层作为表示
#             results = self.model(batch_tokens, repr_layers=[self.model.num_layers], return_contacts=False)

#         # 提取最后一层的 token 表示，形状为 (batch_size, seq_len, embed_dim)
#         token_representations = results["representations"][self.model.num_layers]

#         # 取 token 表示的平均值作为整个序列的 embedding
#         embeddings = token_representations.mean(dim=1)  # shape: (batch_size, embed_dim)

#         return embeddings

class ProteinEncoder(nn.Module):
    def __init__(self, model_name="Rostlab/prot_t5_xl_uniref50", device="cuda", half_precision=True):
        """
        用于编码蛋白质序列的网络，使用ProtT5模型。
        将目标蛋白或 E3 酶的序列输入后输出固定维度的 embedding 向量。

        参数：
          - model_name: 使用的 ProtT5 模型名称
          - device: 设备选择 ("cuda" 或 "cpu")
          - half_precision: 是否使用半精度 (float16)，推荐在GPU上使用以节省显存
        """
        super(ProteinEncoder, self).__init__()
        self.device = device
        self.half_precision = half_precision

        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name)

        if self.device == "cuda" and self.half_precision:
            self.model = self.model.to(torch.float16)

        self.model = self.model.to(self.device)
        self.model.eval()  # 设为评估模式

    def forward(self, sequences):
        processed_sequences = [
            " ".join(list(re.sub(r"[UZOB]", "X", seq[1]))) for seq in sequences
        ]

        inputs = self.tokenizer(processed_sequences,
                                add_special_tokens=True, # 添加 <cls> 和 <eos>
                                padding="longest",       # 填充到批次中的最长序列
                                return_tensors="pt")

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        num_tokens = attention_mask.sum(dim=1, keepdim=True)
        sequence_embeddings = masked_embeddings.sum(dim=1) / num_tokens

        return sequence_embeddings

class ProtacModel(nn.Module):
    def __init__(self, target_encoder, e3_encoder, graph_encoder, smiles_encoder):
        super(ProtacModel, self).__init__()
        self.target_encoder = target_encoder
        self.e3_encoder = e3_encoder
        self.graph_encoder = graph_encoder
        self.smiles_encoder = smiles_encoder
        
        # 蛋白质分支最终输出 320+320=640 维，图分支和 SMILES 分支各 128 维，总计 256 维
        self.fc1 = nn.Linear(2048 + 64 * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, target_seqs, e3_seqs, smiles, graph):
        """
        :param target_seqs: 一个列表，每个元素为元组 (id, target_sequence)
        :param e3_seqs: 一个列表，每个元素为元组 (id, e3_sequence)
        :param smiles: Tensor，形状 (batch, max_len)，表示 PROTAC 分子的 SMILES 数值序列
        :param graph: PyG 的 Batch 对象，表示 PROTAC 分子的图数据
        :return: 输出预测结果，形状 (batch, 2)
        """
        target_inputs = []
        for i, seq in enumerate(target_seqs):
            if not isinstance(seq, str):
                seq = ""
            target_inputs.append(("target_" + str(i), seq))
        e3_inputs = []
        for i, seq in enumerate(e3_seqs):
            if not isinstance(seq, str):
                seq = ""
            e3_inputs.append(("e3_" + str(i), seq))
        
        target_emb = self.target_encoder(target_inputs)  # 输出 (batch, 320)
        e3_emb = self.e3_encoder(e3_inputs)                # 输出 (batch, 320)
        
        # 蛋白质分支输出为两个投影后的向量拼接 (batch, 128)
        protein_emb = torch.cat([target_emb, e3_emb], dim=1)
        
        # 分别编码图数据和 SMILES 数值序列，输出均为 (batch, 64)
        graph_emb = self.graph_encoder(graph)
        smiles_emb = self.smiles_encoder(smiles)
        
        # 拼接三个分支输出，得到 (batch, 256)
        combined = torch.cat([protein_emb, graph_emb, smiles_emb], dim=1)
        
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
