
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from transformers import AdamW
import matplotlib.pyplot as plt 
from SuitData import SuitData,collate_fn_2,collate_fn_1,collate_fn_3
from model import BasicModel,FTModel,MLPModel
from datetime import datetime
import os 

import argparse
def argparser():
    # warning model path should be modified directly in SuitData.py
    parser = argparse.ArgumentParser(description="bert base case level fake law suit discriminator")
    parser.add_argument("--mode",type=str,default="train")
    parser.add_argument("--modelname",type=str,default="DPR")# 这个记得改 
    parser.add_argument("--ckptpath",type=str,default="./model")
    parser.add_argument("--ckptepoch",type=str,default='')
    parser.add_argument("--max_epoch",type=int,default=20)
    parser.add_argument("--device",type=str,default="cuda")
    # in_dim 是dpr输出的维度 
    parser.add_argument("--in_dim",type=int,default=768)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--out_dim",type=int,default=2)
    parser.add_argument("--lr",type=float,default=0.0001)
    return parser.parse_args()



import pickle
def preprocess(args):
    mode = args.mode
    trainset  = SuitData(mode=mode)
    file = f"{args.modelname}_{mode}_feature.pkl"
    if os.path.exists(file):
        content = load_feature(args)
        num_saved = len(content)
    else:
        num_saved = 0 
    trainloader = DataLoader(dataset=trainset,batch_size=1,collate_fn = collate_fn_2,shuffle=False,drop_last=True)
    for iter, (inputs,labels) in tqdm(enumerate(trainloader),total=len(trainloader),colour='green'):
        with open(file,mode='ab') as f:
            pickle.dump({"iter":iter , "labels":labels,"features":inputs},f)
    load_feature(args)

def load_feature(args):
    file = f"{args.modelname}_{args.mode}_feature.pkl"
    content =  []
    with open(file, 'rb') as f:
        while 1:
            try:
                one_pickle_data = pickle.load(f)
                content.append(one_pickle_data)
                #print(one_pickle_data['iter'], end=" ")
            except EOFError:
                print(f"Totally {len(content)} items ")
                break
    return content
if __name__=='__main__':
    args = argparser()
    # python --arch lawformer --mode test 
    # python preprocess.py --modelname tsade15w --mode test
    if args.mode == "b":
        args.mode = 'train'
        preprocess(args)
        args.mode = 'test'
        preprocess(args)
    else:
        preprocess(args)