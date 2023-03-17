
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from transformers import AdamW
import matplotlib.pyplot as plt 
from preprocess import load_feature
from model import BasicModel,FTModel,MLPModel
from datetime import datetime
import os 
class FeatureSet(Dataset):
    def __init__(self,args) -> None:
        super().__init__()
        self.content = load_feature(args)
        self.len = len(self.content)
    def __getitem__(self, index):
        data = self.content[index]
        inputs = data["features"]
        label = data["labels"]
        return inputs,label 
    def __len__(self):
        return self.len 
def collate_feature(data):
    inputs = [i[0] for i in data]
    labels = [int(i[1])  for i in data]
    labels = torch.tensor(labels).long()
    inputs = torch.cat(inputs,dim=0)
    return inputs,labels 
def main(args ,modeltype = "basic"):
    batch_size = args.batch_size
    device = args.device
    epoch= args.max_epoch 
    lr = args.lr
    #datasetpath="C:/Users/ASUS/Desktop/project/FakeLawsuit/dataset/datasets/longtail.txt"
    trainset  = FeatureSet(args)
    trainloader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True,drop_last=True,collate_fn=collate_feature)
    # 这个地方注意 你DPR的输入和输出是多少维度的
    ckpt = f'./model/{args.modelname}_{modeltype}_{args.ckptepoch}th.model'
    if os.path.exists(ckpt):
        FakeDis = torch.load(ckpt)
    elif modeltype=="basic":
        FakeDis = BasicModel(args.in_dim,args.out_dim).to(device)
    elif modeltype == "double":
        FakeDis = FTModel(args.in_dim,args.out_dim).to(device)
    elif modeltype =="MLP":
        FakeDis = MLPModel(args.in_dim,args.out_dim).to(device)
    else:
        raise NotImplementedError(f"Unsupport model type {modeltype}, should be basic or finetune")
    

    for e in range(int(args.ckptepoch),epoch):
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
        print(f"{modeltype} accuracy is ",crt_acc)
        Acc.append(crt_acc)
        torch.save(FakeDis, f'./model/{args.modelname}_{modeltype}_{e}th.model')
        with open("train_log.json",'a',encoding='latin1') as log:
            lf ,rt = "{","}\n"
            string = f"{lf}\"modeltype\":\"{modeltype}\",\"epoch\":\"{e},\"AC\":\"{crt_acc}\",\"NT\":\"{NT}\",\"PT\":\"{PT}\",\"NF\":\"{NF}\",\"PF\":\"{PF}\",\"time\":\"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\"{rt}"
            log.write(string)
import argparse
def argparser():
    # warning model path should be modified directly in SuitData.py
    parser = argparse.ArgumentParser(description="bert base case level fake law suit discriminator")
    parser.add_argument("--mode",type=str,default="train")
    parser.add_argument("--modelname",type=str,default="DPR")# 这个记得改 
    parser.add_argument("--ckptpath",type=str,default="../Representation/model/")
    parser.add_argument("--ckptepoch",type=str,default='0')
    parser.add_argument("--max_epoch",type=int,default=20)
    parser.add_argument("--device",type=str,default="cuda")
    # in_dim 是dpr输出的维度 
    parser.add_argument("--in_dim",type=int,default=768)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--out_dim",type=int,default=2)
    parser.add_argument("--lr",type=float,default=0.0001)
    return parser.parse_args()

# from transformers import AutoModel, AutoTokenizer
# #pretrained_path = 'D:\PreTrainedModels\lawformer'
# pretrained_path = 'path\to\lawformer'
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# lawformer_encoder = AutoModel.from_pretrained(pretrained_path)

# # 这个地方麻烦你自己实现一个集合函数 
# # 这个地方输入的是 文字
# # 对你来说输出的DPR 的特征向量
# # 要是你没有用过这个函数来问我一下
# #dprpath = r'C:\Users\ASUS\Desktop\project\FakeLawsuit\ours\model\DPR'
# dprpath = "path/to/dpr_ckpt"
# DPRModel = AutoModel.from_pretrained(dprpath)
# import os 
# def collate_fn(data):
#     inputs = [i[0] for i in data]
#     labels = [int(i[1])  for i in data]
#     labels = torch.tensor(labels).long()
#     # 上面的你不用管  
#     # 下面的你不充一下
#     # batch_size 是批大小 ，in_dim是dpr的位数
#     # 如果dpr输出的是二维的向量矩阵，那你可以直接按照一个维度相加，或者flatten 
#     # tokens = tokenizer(inputs, return_tensors="pt",truncation=True,max_length=1024)
#     # case_feature = DPRModel(**tokens)['pooler_output']
#     tokens = tokenizer(inputs, return_tensors="pt", padding=True,truncation=True,max_length=512)
#     # for i in tokens:
#     #     print(i,tokens[i].shape)
#     case_feature = DPRModel(**tokens)['pooler_output']
#     #print(case_feature.shape)
#     torch.save(case_feature,f"./DPRfeature/{len(os.listdir('./DPRfeature/'))}.pt")
#     return case_feature,labels


# python main.py --modelname lawformer --mode test     
if __name__=='__main__':
    args = argparser()
    modeltypes = ['basic','double','MLP']
    for modeltype in modeltypes:
        #train DPR 
        main(args,modeltype=modeltype)
    # args.modelname = "lawformer"
    # for modeltype in modeltypes:
    #     #train lawformer
    #     if args.mode == 'train':
    #         main(args,modeltype=modeltype,collate_fn=collate_fn_3)
    #     test_model(f'./model/{modeltype}_{args.max_epoch}.model',collate_fn=collate_fn)



