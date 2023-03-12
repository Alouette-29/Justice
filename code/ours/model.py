import torch
# classifier 
class BasicModel(torch.nn.Module):
    # LF 
    def __init__(self,in_dim=384,out_dim=2) -> None:
        self.fc1 = torch.nn.Linear(in_dim,out_dim)
    def forward(self,embeddings):
        out = self.fc1(embeddings)
        return out   
class FTModel(torch.nn.Module):
    #DH 
    def __init__(self,in_dim=384,out_dim=2) -> None:
        self.fc1 = torch.nn.Linear(in_dim,in_dim//2)
        self.fc2 = torch.nn.Linear(in_dim//2,out_dim)
    def forward(self,embeddings):
        out = self.fc1(embeddings)
        out = self.fc2(out)
        return out 
class MLPModel(torch.nn.Module):
    # DF 
    def __init__(self,in_dim=384,out_dim=2) -> None:
        self.fc1 = torch.nn.Linear(in_dim,in_dim)
        self.fc2 = torch.nn.Linear(in_dim,out_dim)
    def forward(self,embeddings):
        out = self.fc1(embeddings)
        out = self.fc2(out)
        return out 