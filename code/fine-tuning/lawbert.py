import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
pretrained = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
file = open('./cail/competition_stage_2_train.json','r',encoding='utf-8')
import json
line = file.readline()
casedict = json.loads(line)
caseA  =  "".join(casedict['Case_A'])
caseB  = "".join(casedict['Case_B'])
print(len(caseA),len(caseB))
inputs = tokenizer([caseA,caseB], return_tensors="pt",padding="max_length",max_length=300,truncation=True,is_split_into_words=False)
for key in inputs:
    print(key,inputs[key].shape)
pretrained(**inputs).logits.shape
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
class CailModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuneing = False
        self.pretrained = pretrained
        # 2*300*21128
        self.conv2d = torch.nn.Conv2d(in_channels =1,out_channels=1, kernel_size=300,stride=300)
        self.fc = torch.nn.Linear(70,3)
        
    def forward(self, inputs):
        #print(inputs.keys())
        # for i in inputs:
        #     print(inputs[i].size())
        #print(type(inputs))
        caseA  =  "".join(casedict['Case_A'])
        caseB  = "".join(casedict['Case_B'])
        inputA = tokenizer(caseA, return_tensors="pt",padding="max_length",max_length=300,truncation=True,is_split_into_words=False)
        inputB = tokenizer(caseB, return_tensors="pt",padding="max_length",max_length=300,truncation=True,is_split_into_words=False)
        emba = pretrained(**inputA).logits
        embb = pretrained(**inputB).logits
        # 300 * 21128  300*3 
        out = emba+embb
        out = torch.FloatTensor(out)
        out = self.conv2d(out)
        #print(out.shape)
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
lf2classify = CailModel()
lf2classify.fine_tuneing(False)


from transformers import AdamW
#训练
trainset = CailDataset('./cail/competition_stage_2_train.json')
from tqdm import tqdm,trange
def train(epochs):
    lr = 1e-3 if lf2classify.tuneing else 5e-4

    #训练
    optimizer = AdamW(lf2classify.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    lf2classify.train()
    for epoch in range(epochs):
        lf2classify.train()
        step = 0 
        #for step, (inputs, labels) in tqdm(enumerate(trainloader)):
        for i in trange(len(trainset.cases)):
            (inputs, labels) = trainset[i]
            step+=1
            #模型计算
            #print(len(inputs),inputs.keys())
            outs = lf2classify(inputs)
            #print(outs.shape)
            #梯度下降
            outs = outs.squeeze()
            #print(outs,labels)
            one_hot = torch.zeros(3)
            one_hot[labels] = 1
            loss = criterion(outs, torch.FloatTensor(one_hot))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 0:
                print("in epoch "+str(epoch)+" current train loss is "+str(loss.item()))

        torch.save(lf2classify, 'model/lawbert_不微调.model')


print(sum(p.numel() for p in lf2classify.parameters()))
train(10)