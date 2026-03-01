
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import get_encoder

class GCLModel(nn.Module):
    """Graph Contrastive Learning (GCL)模型"""
    def __init__(self, encoder_name, input_dim, hidden_dim, projection_dim, num_layers, dropout=0.5, temperature=0.1):
        super(GCLModel, self).__init__()
        
        self.encoder = get_encoder(encoder_name, input_dim, hidden_dim, num_layers, dropout)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.temperature = temperature

    def forward(self, data1, data2):
        """前向传播，输入为两个增强后的数据视图"""
        # 编码两个视图
        _, z1_pool = self.encoder(data1.x, data1.edge_index, data1.batch)
        _, z2_pool = self.encoder(data2.x, data2.edge_index, data2.batch)
        
        # 通过投影头
        p1 = self.projection_head(z1_pool)
        p2 = self.projection_head(z2_pool)
        
        return p1, p2

    def contrastive_loss(self, p1, p2):
        """计算对比损失 (NT-Xent)"""
        # L2归一化
        p1 = F.normalize(p1, p=2, dim=1)
        p2 = F.normalize(p2, p=2, dim=1)
        # 计算相似度矩阵 (可能为非方阵，如果p1和p2长度不一致)
        sim_matrix = torch.exp(torch.mm(p1, p2.t()) / self.temperature)

        n_rows, n_cols = sim_matrix.size()
        min_n = min(n_rows, n_cols)

        # 如果没有重叠的正样本对，返回0损失以避免错误
        if min_n == 0:
            return torch.tensor(0.0, device=sim_matrix.device)

        # 构建mask：1表示为负样本，0表示正样本位置（仅在对角线上有意义，直到min_n）
        mask = torch.ones_like(sim_matrix)
        idx = torch.arange(min_n, device=sim_matrix.device)
        mask[idx, idx] = 0.0

        # 正样本对的相似度（仅考虑重叠部分）
        pos_sim = sim_matrix[idx, idx]

        # 负样本对的相似度：对每一行求和，但只保留前 min_n 行（对应有正样本的那些行）
        neg_sim_all = (sim_matrix * mask).sum(dim=1)
        neg_sim = neg_sim_all[:min_n]

        # 计算损失（对重叠对求平均）
        loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        return loss.mean()

    def get_embedding(self, data):
        """获取图的嵌入表示（用于下游任务）"""
        self.eval()
        with torch.no_grad():
            _, embedding = self.encoder(data.x, data.edge_index, data.batch)
        return embedding
