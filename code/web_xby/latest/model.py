import torch
from torch import nn 
from torch.nn import  functional as F 
# classifier 
class BasicModel(torch.nn.Module):
    # LF 
    def __init__(self,in_dim=384,out_dim=2) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim,out_dim)
    def forward(self,embeddings):
        out = self.fc1(embeddings)
        return out   
class FTModel(torch.nn.Module):
    #DH 
    def __init__(self,in_dim=384,out_dim=2) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim,in_dim//2)
        self.fc2 = torch.nn.Linear(in_dim//2,out_dim)
    def forward(self,embeddings):
        out = self.fc1(embeddings)
        out = self.fc2(out)
        return out 
class MLPModel(torch.nn.Module):
    # DF 
    def __init__(self,in_dim=384,out_dim=2) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim,in_dim)
        self.fc2 = torch.nn.Linear(in_dim,out_dim)
    def forward(self,embeddings):
        out = self.fc1(embeddings)
        out = self.fc2(out)
        return out 
class SublayerConnection(nn.Module):
    '''残差网络'''
    def __init__(self,embed_size,dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,layer):
        return x + self.dropout(layer(self.norm(x)))

class FFN(nn.Module):
    '''Position-wise Feed-Forward Network'''
    def __init__(self,embed_size,expansion_rate,dropout):
        super().__init__()
        self.feed_forward = nn.Sequential(
        nn.Linear(embed_size,embed_size*expansion_rate),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(embed_size*expansion_rate,embed_size)
        )
    def forward(self,x):
        return self.feed_forward(x)

class AttenModel(nn.Module):
    def __init__(self,hidden_size=384) -> None:
        super().__init__()
        self.embed = nn.Linear(hidden_size,hidden_size)
        #self.add_norm1 = SublayerConnection(hidden_size,0.1)
        self.attention = nn.MultiheadAttention(hidden_size,4,batch_first=True)
        self.add_norm2 = SublayerConnection(hidden_size,0.1)
        self.ffn = FFN(hidden_size,4,0.1)
        self.add_norm3 = SublayerConnection(hidden_size,0.1)
        self.linear = nn.Linear(hidden_size,2)

    def forward(self,x):
        x = self.embed(x)
        atten = self.attention(x,x,x)[0]
        out = self.ffn(atten)
        atten = self.add_norm2(x,lambda x: self.attention(x,x,x)[0])
        out = self.add_norm3(atten, self.ffn)
        return self.linear(out)
    