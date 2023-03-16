import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from transformers import AdamW
import matplotlib.pyplot as plt 
from SuitData import SuitData,collate_fn_1,collate_fn_2,collate_fn_3
from model import BasicModel,FTModel,MLPModel
from datetime import datetime
import os 
def argparser():
    # warning model path should be modified directly in SuitData.py
    parser = argparse.ArgumentParser(description="bert base case level fake law suit discriminator")
    parser.add_argument("--modelname",type=str,default="DPR")# 这个记得改 
    parser.add_argument("--ckptpath",type=str,default="./model")
    parser.add_argument("--ckptepoch",type=str,default='10')
    parser.add_argument("--device",type=str,default="cuda")
    # in_dim 是dpr输出的维度 
    return parser.parse_args()
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
if __name__=='__main__':
    args = argparser()
    modeltypes = ['basic','double','MLP']
    # python test.py --modelname tsade15w --ckptepoch 10 
    for modeltype in modeltypes:
        test_model(f'./model/{args.modelname}_{modeltype}_{args.ckptepoch}th.model',collate_fn=collate_fn_2)