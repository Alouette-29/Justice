import argparse
import torch
from  datetime import datetime
from tqdm import tqdm 
from SuitData import SuitData
from preprocess import load_feature
def argparser():
    # warning model path should be modified directly in SuitData.py
    parser = argparse.ArgumentParser(description="bert base case level fake law suit discriminator")
    parser.add_argument("--modelname",type=str,default="DPR")# 这个记得改 
    parser.add_argument("--ckptpath",type=str,default="./model")
    parser.add_argument("--ckptepoch",type=str,default='20')
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--mode",type=str,default="test")
    # in_dim 是dpr输出的维度 
    return parser.parse_args()

def test_model(args,device = "cuda"):
    model = f'{args.ckptpath}/{args.modelname}_{modeltype}_{args.ckptepoch}th.model'
    model = torch.load(model).to(device)
    dataset = load_feature(args)
    correct = 0 
    for sample in tqdm(dataset):
        inputs = sample['features']
        labels = sample['labels']
        inputs.to(device)
        labels = labels.to(device)
        outs = model(inputs)
        fake = torch.argmax(outs,axis=1)
        correct += torch.sum(fake==labels)
    with open("test_log.json",'a',encoding='latin1') as log:
        lf ,rt = "{","}\n"
        string = f"{lf}\"modeltype\":\"{modeltype}\",\"modelname\":\"{args.modelname},\"epoch\":\"{args.ckptepoch},\"AC\":\"{correct/len(dataset)}\",\"time\":\"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\"{rt}"
        log.write(string)
    print("ACCURACY in testset is :" , correct/len(dataset))
if __name__=='__main__':
    args = argparser()
    modeltypes = ['basic','double','MLP']
    # python test.py --modelname tsade15w --ckptepoch 10 
    for modeltype in modeltypes:
        test_model(args)