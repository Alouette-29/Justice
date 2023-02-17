# embedding 模块要单独做
# 然后连接层往下接 
# 先用sentence transformer 
# 再用bert 

from sentence_transformers import SentenceTransformer
pretrained = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("parameters: ",sum(i.numel() for i in pretrained.parameters()))
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm,trange
import json
from transformers import AdamW
class EssenceSet(Dataset):
    def __init__(self,path,labels=None) -> None:
        super().__init__()
        self.path = path
        if path.endswith("pt")==False and path.endswith("pth")==False:
            # given a txt file type
            file = open(path,'r',encoding='utf-8')
            self.cases = file.readlines()
            self.cases = [json.loads(item) for item in self.cases]
            file.close()
        else:
            tensors = torch.load(path)
            labels = torch.load(labels)
            self.cases = [{'sentence':tensor,'label':label} for (tensor,label) in zip(tensors,labels)]
        self.length = len(self.cases)
        print("the length of cases is",self.length)
    def __getitem__(self, index) :
        item = self.cases[index]
        sentence = item['sentence']
        label = item['label']
        return sentence, label
    def __len__(self):
        return self.length 

class EssenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(384,384)
        self.fc2 = torch.nn.Linear(384,2)
    def forward(self,embeddings):
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
    sentences = [i[0] for i in data]
    labels = [int(i[1]) for i in data]

    # get embeedings 
    embeddings = sentenceformer(sentences)

    labels = torch.Tensor(labels).long()

    return embeddings,labels
tensor_path = "../cail/tensor_classify_set.pth"
label_path = "../cail/label_classify_set.pth"
dataset_path = "../cail/classify_set.json"
trainset = EssenceSet(path=tensor_path,labels=label_path)
batch_size = 64
epoch = 20
#trainloader = DataLoader(trainset,batch_size,collate_fn=collate_fn,shuffle=True)
trainloader = DataLoader(trainset,batch_size,shuffle=True)
CE_loss = torch.nn.CrossEntropyLoss()
model = EssenceModel()
lr = 1e-5 
optimizer = AdamW(model.parameters(), lr=lr)

torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1,verbose=True)
length = len(trainset.cases)

device  = "cuda:0"
model = torch.load("19th_essence_predict_model.model")
model.to(device=device)
for i in range(20,40):
    losses = 0 
    correct = 0 
    for step,(embeddings,labels) in tqdm(enumerate(trainloader),total=len(trainloader),colour='green'):
        #print(embeddings.shape,labels.shape)
        outs = model(embeddings)
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
    if i%5==0 or i == epoch-1:
        torch.save(model,str(i)+"th_relation_predict_model.model")
