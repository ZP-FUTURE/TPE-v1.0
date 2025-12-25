import torch
import torch.nn as nn
import math

# ==========================================
# 1. 初始化工具 
# ==========================================
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """
    JAX-style truncated normal initialization.
    PyTorch 官方的 trunc_normal_ 标准差计算有偏差，这里手动实现以保证深层网络初始化稳定。
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            
            # 计算补偿后的标准差
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor

# ==========================================
# 2. 增强版 Transformer 模型结构
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class AttentionPooling(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        w = self.attention(x)         
        x = torch.sum(x * w, dim=1)   
        return x

class PendulumTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg['input_dim']
        d_model = cfg['model_dim']
        output_dim = cfg['output_dim']
       
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg['num_heads'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout'],
            activation='gelu',   
            batch_first=True,
            norm_first=True     
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=cfg['num_layers']
        )
        
        self.pooling = AttentionPooling(d_model)
        
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            
            nn.Linear(d_model // 2, output_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x) 
        
        x = self.pooling(x) 
        out = self.regressor(x) 
        return out