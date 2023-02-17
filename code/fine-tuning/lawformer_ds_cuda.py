from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
lawformer = AutoModel.from_pretrained("thunlp/Lawformer")
import torch
import json
class CailDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path 
        file = open(path,'r',encoding='utf-8')
        cases = file.readlines()
        self.cases = [json.loads(i) for i in cases]
        self.length = len(self.cases)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.cases[index]
        caseA = "".join(data['Case_A'])
        caseB = "".join(data['Case_B'])
        label = data['label']
        inputs = tokenizer([caseA,caseB], return_tensors="pt",padding="max_length",max_length=300,truncation=True)
        return inputs,label

def collate_fn(data):
    tensors = [i[0] for i in data]
    labels = [i[1]  for i in data]
    inputs_ids = torch.cat([i['input_ids'] for i in tensors])
    token_type_ids = torch.cat([i['token_type_ids'] for i in tensors])
    mask = torch.cat([i['attention_mask'] for i in tensors])
    inputs = {"input_ids":inputs_ids,"token_type_ids":token_type_ids,"attention_mask":mask}
    return inputs,labels

trainset = CailDataset('./cail/competition_stage_2_train.json')
#数据加载器
trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                     batch_size=1,
                                     collate_fn = collate_fn,
                                     shuffle=True,
                                     drop_last=True)



#模型试算
#[b, lens] -> [b, lens, 768]
#pretrained(**inputs).last_hidden_state.shape
import torch  
class Lawfomer2Super(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuneing = False
        self.pretrained = lawformer
        # 300 is max length of lawformer 
        self.conv1d = torch.nn.Conv1d(in_channels=300,out_channels=1,kernel_size=1)
        self.conv2d  = torch.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=(2,1),stride=(2,1))
        self.fc = torch.nn.Linear(768, 3)

    def forward(self, inputs):
        #print(inputs.keys())
        # inputs : batch_size * max_length * 768 
        # conv1d output : batch_size * 1 * 768
        # result : batch_size*1 * 3 
        if self.tuneing:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = self.pretrained(**inputs).last_hidden_state
        print(out.shape)
        
        out = self.conv1d(out)
        out = out.squeeze()
        out = out.unsqueeze(0)
        print(out.shape)
        out = self.conv2d(out)
        print(out.shape)
        out = self.fc(out)
        out = out.squeeze()
        print(out.shape)
        
        return out

    def fine_tuneing(self, tuneing):
        self.tuneing = tuneing
        if tuneing:
            for i in self.pretrained.parameters():
                i.requires_grad = True

            lawformer.train()
            self.pretrained = lawformer
        else:
            for i in lawformer.parameters():
                i.requires_grad_(False)

            lawformer.eval()
            #print("any")
            self.pretrained = None

lf2classify = Lawfomer2Super()
lf2classify.fine_tuneing(tuneing=True)
device = "cuda:0"
lf2classify.to(device=device)
lf2classify.pretrained.to(device)
from transformers import AdamW
#训练
from tqdm import tqdm
def train(epochs,device):
    lr = 2e-4 if lf2classify.tuneing else 5e-4

    #训练
    optimizer = AdamW(lf2classify.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    lf2classify.train()
    for epoch in range(epochs):
        lf2classify.train()
        step = 0
        correct = 0
        for step, (inputs,labels) in tqdm(enumerate(trainloader),total=len(trainloader),colour='green'):
            #print("labels",labels)
            step+=1
            #模型计算
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            outs = lf2classify(inputs)
            #梯度下降
            outs = outs.squeeze()
            labels = torch.tensor(labels).long()
            labels = labels.to(device)
            print("cretirion",outs,labels)
            loss = criterion(outs,labels)
            fake = torch.argmax(outs)
            
            correct += fake==labels
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("in epoch "+str(epoch)+" current train loss is "+str(loss.item()))
        print("accuracy is ",correct/len(trainset.cases))

        torch.save(lf2classify, 'model/lawformer_微调.model')

train(10,device)
print(sum(p.numel() for p in lf2classify.parameters()))