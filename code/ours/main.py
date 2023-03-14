
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from transformers import AdamW
import matplotlib.pyplot as plt 
from SuitData import SuitData,collate_fn_1,collate_fn_2,collate_fn_3
from model import BasicModel,FTModel,MLPModel
from datetime import datetime
def main(args ,modeltype = "basic",collate_fn=collate_fn_1):
    batch_size = args.batch_size
    device = args.device
    epoch= args.max_epoch 
    lr = args.lr
    #datasetpath="C:/Users/ASUS/Desktop/project/FakeLawsuit/dataset/datasets/longtail.txt"
    trainset  = SuitData()
    trainloader = DataLoader(dataset=trainset,batch_size=batch_size,collate_fn = collate_fn,shuffle=True,drop_last=True)
    # 这个地方注意 你DPR的输入和输出是多少维度的
    if modeltype=="basic":
        FakeDis = BasicModel(args.in_dim,args.out_dim).to(device)
    elif modeltype == "double":
        FakeDis = FTModel(args.in_dim,args.out_dim).to(device)
    elif modeltype =="MLP":
        FakeDis = MLPModel(args.in_dim,args.out_dim).to(device)
    else:
        raise NotImplementedError(f"Unsupport model type {modeltype}, should be basic or finetune")
    

    for e in range(epoch):
        correct = 0
        optimizer = AdamW(FakeDis.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        Acc = []
        NT , NF,PT,PF =0,0,0,0 # negative true negative false  positive true positive false 
        ne_sample , po_sample =0,0
        for iter, (inputs,labels) in tqdm(enumerate(trainloader),total=len(trainloader),colour='green'):
            #print("labels",labels)
            #模型计算
            inputs = inputs.to(device)
            #print(inputs.device)
            outs = FakeDis(inputs)
            #梯度下降
            outs = outs.squeeze()
            labels = labels.to(device)
            #print("cretirion",outs,labels)
            loss = criterion(outs,labels)
            fake = torch.argmax(outs,axis=1)
            # 计算准确率还要计算召回率
            correct += torch.sum(fake==labels)
            NT+=torch.sum((fake==0)==(labels==fake))
            NF+=torch.sum((fake==1)==(labels!=fake))
            PF+=torch.sum((fake==0)==(labels!=fake))
            PT+=torch.sum((fake==1)==(labels==fake))
            ne_sample+=torch.sum(labels==0)
            po_sample+=torch.sum(labels==1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print("in epoch "+str(e)+" current train loss is "+str(loss.item()))
        crt_acc = ((correct/trainset.len).cpu().detach())
        print(f"{modeltype}accuracy is ",crt_acc)
        Acc.append(crt_acc)
        torch.save(FakeDis, f'./model/{args.modelname}{modeltype}_{e}th.model')
        with open("train_log.json",'a',encoding='latin1') as log:
            lf ,rt = "{","}\n"
            string = f"{lf}\"modeltype\":\"{modeltype}\",\"epoch\":\"{e},\"AC\":\"{crt_acc}\",\"NT\":\"{NT}\",\"PT\":\"{PT}\",\"NF\":\"{NF}\",\"PF\":\"{PF}\",\"time\":\"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\"{rt}"
            log.write(string)

def test_model(model='model/basic.model',batch_size = 16 ,device = "cuda",collate_fn=collate_fn_1):
    model = torch.load(model).to(device)
    testset = SuitData(mode = "test")
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                     batch_size=batch_size,
                                     collate_fn = collate_fn,
                                     shuffle=False,
                                     drop_last=True)
    correct = 0 
    for iter,(inputs,labels) in tqdm(enumerate(testloader),total = len(testloader)):
        inputs.to(device)
        labels = labels.to(device)
        outs = model(inputs)
        fake = torch.argmax(outs,axis=1)
        correct += torch.sum(fake==labels)
        if iter==1000:
            break
    print("ACCURACY in testset is :" , correct/testset.len)

import argparse
def argparser():
    # warning model path should be modified directly in SuitData.py
    parser = argparse.ArgumentParser(description="bert base case level fake law suit discriminator")
    parser.add_argument("--mode",type=str,default="train")
    parser.add_argument("--modelname",type=str,default="TSADE")# 这个记得改 
    parser.add_argument("--ckptpath",type=str,default="./model")
    parser.add_argument("--ckptepoch",type=int,default=10)
    parser.add_argument("--max_epoch",type=int,default=20)
    parser.add_argument("--device",type=str,default="cuda")
    # in_dim 是dpr输出的维度 
    parser.add_argument("--in_dim",type=int,default=768)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--out_dim",type=int,default=2)
    parser.add_argument("--lr",type=float,default=0.0001)
    return parser.parse_args()

from transformers import AutoModel, AutoTokenizer
#pretrained_path = 'D:\PreTrainedModels\lawformer'
pretrained_path = 'path\to\lawformer'
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
lawformer_encoder = AutoModel.from_pretrained(pretrained_path)

# 这个地方麻烦你自己实现一个集合函数 
# 这个地方输入的是 文字
# 对你来说输出的DPR 的特征向量
# 要是你没有用过这个函数来问我一下
#dprpath = r'C:\Users\ASUS\Desktop\project\FakeLawsuit\ours\model\DPR'
dprpath = "path/to/dpr_ckpt"
DPRModel = AutoModel.from_pretrained(dprpath)
import os 
def collate_fn(data):
    inputs = [i[0] for i in data]
    labels = [int(i[1])  for i in data]
    labels = torch.tensor(labels).long()
    # 上面的你不用管  
    # 下面的你不充一下
    # batch_size 是批大小 ，in_dim是dpr的位数
    # 如果dpr输出的是二维的向量矩阵，那你可以直接按照一个维度相加，或者flatten 
    # tokens = tokenizer(inputs, return_tensors="pt",truncation=True,max_length=1024)
    # case_feature = DPRModel(**tokens)['pooler_output']
    tokens = tokenizer(inputs, return_tensors="pt", padding=True,truncation=True,max_length=512)
    # for i in tokens:
    #     print(i,tokens[i].shape)
    case_feature = DPRModel(**tokens)['pooler_output']
    #print(case_feature.shape)
    torch.save(case_feature,f"./DPRfeature/{len(os.listdir('./DPRfeature/'))}.pt")
    return case_feature,labels


def preprocess(args):
    batch_size = args.batch_size
    trainset  = SuitData()
    os.makedirs('./DPRfeature/',exist_ok=True)
    trainloader = DataLoader(dataset=trainset,batch_size=batch_size,collate_fn = collate_fn,shuffle=False,drop_last=True)
    for iter, (inputs,labels) in tqdm(enumerate(trainloader),total=len(trainloader),colour='green'):
        continue
if __name__=='__main__':
    args = argparser()
    modeltypes = ['basic','double','MLP']
    for modeltype in modeltypes:
        #print(args.mode)
        if args.mode == 'train':
            main(args,modeltype=modeltype,collate_fn=collate_fn)
        test_model(f'./model/{modeltype}_{args.max_epoch}.model',collate_fn=collate_fn)



