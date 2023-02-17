# embedding 模块要单独做
# 然后连接层往下接 
# 先用sentence transformer 
# 再用bert 
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
pretrained = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("parameters: ",sum(i.numel() for i in pretrained.parameters()))
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm,trange
import json
from transformers import AdamW
class RelationSet(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.path = path
        file = open(path,'r',encoding='utf-8')
        self.cases = file.readlines()
        file.close()
        self.length = len(self.cases)
        print("the length of cases is ",self.length)
    def __getitem__(self, index) :
        item = self.cases[index]
        item = json.loads(item)
        s1 = item['s1']
        s2 = item['s2']
        label = item['label']
        return [s1,s2], label
    def __len__(self):
        return self.length 

class RelationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(768,384)
        self.fc2 = torch.nn.Linear(384,2)
    def forward(self,embeddings):
        # these two should have equal 

        out = self.fc1(embeddings)
        out = self.fc2(out)
        return out 


def sentenceformer(data):
    embeeddings = pretrained.encode(data,convert_to_tensor=True)
    return embeeddings
def bertchinese():
    pass
def lawformer():
    pass 
def collate_fn(data):
    s1s = [i[0][0] for i in data]
    s2s = [i[0][1] for i in data]
    labels = [int(i[1]) for i in data]

    # get embeedings 
    embedding1 = sentenceformer(s1s)
    embedding2 = sentenceformer(s2s)
    labels = torch.Tensor(labels).long()
    embedding = torch.cat([embedding1,embedding2],axis = 1)
    #print(embedding.shape)
    return embedding,labels

dataset_path = "../cail/sentence_relation_set.json"
trainset = RelationSet(dataset_path)
batch_size = 16
epoch = 10
trainloader = DataLoader(trainset,batch_size,collate_fn=collate_fn,shuffle=True)
CE_loss = torch.nn.CrossEntropyLoss()
model = RelationModel()
lr = 1e-3
optimizer = AdamW(model.parameters(), lr=lr)

torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1,verbose=True)
length = len(trainset.cases)

device  = "cuda:0"
model.to(device=device)
for i in range(epoch):
    losses = 0 
    correct = 0 
    for step,(embedding,labels) in tqdm(enumerate(trainloader),total=len(trainloader),colour='green'):
        embedding = embedding.to(device)
        labels = labels.to(device)
        outs = model(embedding)
        #print(outs.shape)
        loss = CE_loss(outs,labels)
        losses+=loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #print(torch.argmax(outs,axis = 1).shape)
       
        correct+=sum(torch.argmax(outs,axis = 1)==labels)
    print('epoch '+str(i)+'loss',losses/length)
    print(correct/length)
    if i%20==0 or i == epoch-1:
        torch.save(model,str(i)+"th_relation_predict_model.model")
