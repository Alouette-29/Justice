import torch
import json
import numpy as np 

class EssenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(384,384)
        self.fc2 = torch.nn.Linear(384,2)
    def forward(self,embeddings):
        out = self.fc1(embeddings)
        out = self.fc2(out)
        return out 
from sentence_transformers import SentenceTransformer
sentenceformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("sentence_former parameters: ",sum(i.numel() for i in sentenceformer.parameters()))



def sentenceformer_emb(sentences,device):
    embeeddings = sentenceformer.encode(sentences,convert_to_tensor=True)
    return embeeddings
# 输入文本或者输入一个文件地址 

device = "cuda:0"
example  = ' 湖南省浏阳市人民法院执行通知书（2021）湘0181执2137号曾德贵：你（单位）与李海文民间借贷纠纷一案，本院作出的（2020）湘0181民初8068号民事调解书已发生法律效力。申请执行人李海文于2021年4月6日向本院申请强制执行，本院2021年4月6日依法立案执行。依照《中华人民共和国民事诉讼法》第二百四十条的规定，责令你（单位）于2021年4月19日之前履行下列义务：1、向申请人李海文支付案款26500.0元。2、负担案件申请执行费。开户银行：长沙银行联城支行户名：浏阳市人民法院案款专户账号：800125239611019交款时应注明交款人姓名或名称、被执行人姓名或名称以及交款用途。特此通知。二〇二一年四月六日承办法官：袁昆联系电话：18817136709本院地址：浏阳市环府路1号邮编：410300风险提示：根据《最高人民法院关于公布失信被执行人名单信息的若干规定》第一条的规定，被执行人未履行生效法律文书确定的义务，并具有下列情形之一的，人民法院应当将其纳入失信被执行人名单，依法对其进行信用惩戒：（一）有履行能力而拒不履行生效法律文书确定义务的；（二）以伪造证据、暴力、威胁等方法妨碍、抗拒执行的；（三）以虚假诉讼、虚假仲裁或者以隐匿、转移财产等方法规避执行的；（四）违反财产报告制度的；（五）违反限制消费令的；（六）无正当理由拒不履行执行和解协议的。'

# 对于sentenceformer
sentences = example.split("。")
# 对于lawformer

#先做文本分段
model_path = "./check_dup/40th_claasify.model"
classify = EssenceModel()
classify = torch.load(model_path)
#找到标号为2的句子
emb = sentenceformer_emb(sentences,device)
print(emb.shape)
facts_index = classify(emb)
# print(facts_index.shape)
# print(facts_index)
# print(torch.argmax(facts_index[0],axis=0))
facts_sentences = [sentences[i] for i in range(len(facts_index)) if torch.argmax(facts_index[i])==1] #是不是2呢 
# print(torch.argmax(facts_index,axis=1))
facts_emb = emb[torch.argmax(facts_index,axis=1)==1].T # 384 *m
# 查重匹配 
print(facts_emb)
database_path = "./subset/ensence_tensor.pth"
sentence_base_path = "./subset/essence_sentence.json"
with open(sentence_base_path,'r',encoding='utf-8') as base:
    sentence_base = base.readlines()[:1000]
sentence_base = [json.loads(i) for i in sentence_base]
sentence_base = [i['sentence'] for i in sentence_base]
# shape : n * 384 
database = torch.load(database_path)[:1000]
print(database)
#basel2 = torch.sqrt(torch.sum(torch.square(database),axis = 0))
#print(database.shape,basel2.shape)
#database = database/basel2

# query matrix
# factsl2 = torch.sqrt(torch.sum(torch.square(facts_emb),axis = 1))
#factsl2 = 
#print(facts_emb.shape,factsl2.shape)
#facts_emb = (facts_emb.T/factsl2).T
#print(facts_emb)
#print(database)

# dot product represent similarity , equal to cosine similarity 


def cos_sim(a, b):
    a_norm = np.linalg.norm(a,axis=1)
    b_norm = np.linalg.norm(b,axis=0)
    a =( a.T/a_norm).T
    b = b/b_norm
    cos = np.dot(a,b)
    #cos = np.dot(a,b)/(a_norm * b_norm)
    return cos
#print(cos_sim(database.cpu(),facts_emb.cpu()).shape)

#print(torch.max(facts_emb),torch.max(database))
similarity_matrix = np.array((database @ facts_emb).to('cpu'))
similarity_matrix = cos_sim(database.cpu(),facts_emb.cpu())

print("similar ",similarity_matrix.shape)
import sys
sys.path.append("../")
from utensil.painting import heatmap

columns = [i for i in range(similarity_matrix.shape[0])]
rows = [i for i in range(similarity_matrix.shape[1])]
im,cbar=heatmap(similarity_matrix,columns,rows)
#print(similarity_matrix)
# give all the sentences pairs if similarity > 95% 
print("given sentences                         match sentences")
print(np.max(similarity_matrix))
for i in range(similarity_matrix.shape[0]):
    for j in range(similarity_matrix.shape[1]):
        if similarity_matrix[i][j]>0.80:
            print(sentence_base[i],'[SEP]',facts_sentences[j])
            pass


