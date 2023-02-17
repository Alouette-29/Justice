import torch 
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
pretrained = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("parameters: ",sum(i.numel() for i in pretrained.parameters()))
device = 'cuda:0'
import json 
from torch.utils.data import Dataset,DataLoader
class CailDataset(Dataset):
    def __init__(self,dataset) -> None:
        super(CailDataset).__init__()
        self.dataset = dataset
        file = open(dataset,'r',encoding='utf-8')
        cases = file.readlines()
        self.cases = [json.loads(i) for i in cases]
        self.length = len(self.cases)
    def __getitem__(self, index) :
        # caseA = self.cases[index]["Case_A"]
        # caseB = self.cases[index]["Case_B"]
        # label = self.cases[index]["label"]
        # return caseA,caseB,label
        return index 
    def __len__(self):
        return self.length

#模型试算
#[b, lens] -> [b, lens, 768]
#pretrained(**inputs).last_hidden_state.shape
from sentence_transformers import SentenceTransformer
pretrained = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
class CailModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuneing = False
        self.pretrained = pretrained
        self.fc = torch.nn.Linear(384,3)

    def forward(self, inputs):
        #print(inputs.keys())
        # for i in inputs:
        #     print(inputs[i].size())
        #print(type(inputs))
        emba = pretrained.encode("".join(inputs['Case_A']))
        embb = pretrained.encode("".join(inputs['Case_B']))
        out = emba+embb
        out = torch.FloatTensor(out)
        out = self.fc(out)
        #out = torch.argmax(out).long()
        return out

    def fine_tuneing(self, tuneing):
        self.tuneing = tuneing
        if tuneing:
            for i in pretrained.parameters():
                i.requires_grad = True

            pretrained.train()
            self.pretrained = pretrained
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)

            pretrained.eval()
            self.pretrained = None
model = CailModel()

path = './cail/competition_stage_2_train.json'
file = open(path,'r',encoding='utf-8')
cases = file.readlines()
cases = [json.loads(i) for i in cases]
file.close()
epoch  = 100 
from transformers import AdamW
lr = 2e-5
optimizer = AdamW(model.parameters(), lr=lr)
from torch import nn 
length = len(cases)
from tqdm import trange 
CE_loss = nn.CrossEntropyLoss()     #定义损失函数


for i in range(epoch):
    losses  = 0 
    correct = 0 
    for index  in trange(length,colour='green'):
        output  = model(cases[index])
        label   = torch.tensor(cases[index]['label'])
        loss = CE_loss(output,label)
        losses+=loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if torch.argmax(output)==label:
            correct+=1
    print('epoch '+str(i)+'loss',losses/length)
    print(correct/length)
    if i%20==0 or i == epoch-1:
        torch.save(model,str(i)+"th_model.model")
