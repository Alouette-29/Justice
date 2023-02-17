import torch
import json 
from transformers import AutoModel, AutoTokenizer

# lawformer  
class CailDataset(torch.utils.data.Dataset):
    def __init__(self, path,tokenizer):
        self.path = path 
        file = open(path,'r',encoding='utf-8')
        cases = file.readlines()
        self.cases = [json.loads(i) for i in cases]
        self.length = len(self.cases)
        self.tokenizer  =  tokenizer
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.cases[index]
        caseA = "".join(data['Case_A'])
        caseB = "".join(data['Case_B'])
        label = data['label']
        inputs = self.tokenizer([caseA,caseB], return_tensors="pt",padding="max_length",max_length=300,truncation=True)
        return inputs,label

def collate_fn(data):
    tensors = [i[0] for i in data]
    labels = [i[1]  for i in data]
    inputs_ids = torch.cat([i['input_ids'] for i in tensors])
    token_type_ids = torch.cat([i['token_type_ids'] for i in tensors])
    mask = torch.cat([i['attention_mask'] for i in tensors])
    inputs = {"input_ids":inputs_ids,"token_type_ids":token_type_ids,"attention_mask":mask}
    return inputs,labels