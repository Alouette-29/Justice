#dataset_path = "../cail/classify_set.json"
#tensor_path = "../cail/tensor_classify_set.pth"
#label_path = "../cail/label_classify_set.pth"
dataset_path = "./文本分段/fake_facts.json"
tensor_path = "./tensor_fake_facts.pth"
label_path = "./label_fake_facts.pth"

print("check ")
import json 
import torch
file = open(dataset_path,'r',encoding='utf-8')
cases = file.readlines()
file.close()
length = len(cases)
print("the length of cases is",length)

cases  = [json.loads(item) for item in cases]
sentences = [item['sentence'] for item in cases]
labels = [int(item['label '] )for item in cases]

from sentence_transformers import SentenceTransformer
pretrained = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("parameters: ",sum(i.numel() for i in pretrained.parameters()))



def sentenceformer(data):
    embeeddings = pretrained.encode(data,convert_to_tensor=True)
    return embeeddings

embeddings = sentenceformer(sentences).to("cuda:0")
print("shape of the embbding",embeddings.shape)
torch.save(embeddings,tensor_path)
labels = torch.tensor(labels).to("cuda:0")
torch.save(labels,label_path)


