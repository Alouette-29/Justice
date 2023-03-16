import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np 
import pickle 
import os  
from sentence_transformers import SentenceTransformer

datasetpath="C:/Users/ASUS/Desktop/project/FakeLawsuit/dataset/datasets/valset.txt"
#datasetpath="C:/Users/ASUS/Desktop/project/FakeLawsuit/dataset/datasets/longtail.txt"
# 这个函数你不用管，是用来划分数据集的，我已经把划分好的传上去了 
def split_val_test(path="path/to/valset.txt",proportion = 0.1):
    filename =  os.path.basename(path).split('.')[-2]
    with open(path,'r',encoding='utf-8') as f:
        content = f.readlines()
    content  =  [json.loads(i) for i in content]
    # path = "path/to/valset"
    if os.path.exists(filename+"_idx.pkl"):
        diction = pickle.load(open(filename+"_idx.pkl",'rb'))
        testidx = diction['test']
        trainidx = diction['train']
        return content, testidx,trainidx
        
    length = len(content)
    #得到关于全集的一个划分 
    division = {"True":[],"False":[]}
    for i in range(length):
        if content[i]['label']==0:
            division['False'].append(i)
        else:
            division['True'].append(i)
    #随机采样
    nT = len(division['True'])
    nF = len(division['False'])
    Tidx = np.random.randint(0,nT,size=int(nT*proportion))
    Fidx = np.random.randint(0,nF,size=int(nF*proportion))
    # 小样本数据
    testset ={"True":[division['True'][i] for i in Tidx],"False":[division['False'][i] for i in Fidx]}
    testidx = testset["True"]+testset['False']
    trainidx = list(set(range(length))-set(testidx))
    testidx = np.array(testidx)
    trainidx = np.array(trainidx)
    pickle.dump({"filepath":path,"test":testidx,"train":trainidx},open(filename+"_idx.pkl",'wb'))
    #pickle.dump({"filepath":path,"test":testidx,"train":trainidx},filename)
    return content , trainidx , testidx


class SuitData(Dataset):
    def __init__(self,path=datasetpath,mode  = "train") -> None:
        super().__init__()
        self.content , trainidx , testidx = split_val_test(path)
        if mode == 'train':
            self.len = len(trainidx)
            self.index = trainidx
        elif mode == 'test':
            self.len = len(testidx)
            self.index = testidx
        else:
            print("warning: unknown type of dataset , using train for default ")
            self.len = len(trainidx)
            self.index = trainidx       
    def __getitem__(self, index):
        data = self.content[self.index[index]]
        inputs = data["content"]
        label = data["label"]
        #inputs = tokenizer([inputs], return_tensors="pt",padding="max_length",max_length=300,truncation=True)
        return inputs,label 
    def __len__(self):
        return self.len 
dataset = SuitData(datasetpath)
base_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def collate_fn_1(data):
    inputs = [i[0] for i in data]
    labels = [int(i[1])  for i in data]
    labels = torch.tensor(labels).long()
    inputs = base_encoder.encode(inputs,convert_to_tensor=True)
    return inputs,labels
model_path = r'./output/150000'
TBADE_encoder = SentenceTransformer(model_path)
def collate_fn_2(data):
    inputs = [i[0] for i in data]
    labels = [int(i[1])  for i in data]
    labels = torch.tensor(labels).long()
    inputs = TBADE_encoder.encode(inputs,convert_to_tensor=True)
    return inputs,labels
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
#pretrained_path = 'D:\PreTrainedModels\lawformer'
#pretrained_path = "path/to/lawformer"
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
lawformer_encoder = AutoModel.from_pretrained("thunlp/lawformer")
def collate_fn_3(data):
    inputs = [i[0] for i in data]
    labels = [int(i[1])  for i in data]
    labels = torch.tensor(labels).long()
    tokens = tokenizer(inputs, return_tensors="pt", padding=True,truncation=True,max_length=1024 )
    case_feature = lawformer_encoder(**tokens)['pooler_output']
    # once for a case 
    case_feature = case_feature.detach()
    return case_feature,labels