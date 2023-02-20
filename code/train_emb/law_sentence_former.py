from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample,models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


import torch 
from torch.utils.data import Dataset,DataLoader
from datetime import datetime  


import os , json 
"""
要进行句子级别的文本嵌入，
必要用一个词嵌入模型来提取文本特征
然后用一个池化层来对其向量
然后用一个线性层来任意调节向量长度  
"""
device  = "cuda:0"
# 定义 词嵌入的模型
word_embedding_model = models.Transformer('bert-base-chinese', max_seq_length=256)
# 定义池化层
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# 最后组合模型 
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model.to(device)

class downstramset(Dataset):
    def __init__(self,data,sample=-1) -> None:
        super().__init__()
        # 支持两种传入方式
        # 第一种是列表组成的字典
        if type(data)==list and len(data) and type(data[0])==dict:
            self.data = data
        # 第二种是 按照指定长度读取文件
        if type(data)==str and os.path.exists(data) :
            file = open(data,'r',encoding='utf-8') 

            if sample==-1:
                self.data = file.readlines()
                self.data = [json.loads(i) for i in self.data]
            elif sample>0:
                self.data = []
                for i in range(sample):
                    self.data.append(json.loads(file.readline()))
            file.close()
    def __getitem__(self, index) :
        item =self.data[index]
        s1 = item['s1']
        s2 = item['s2']
        score = float(item['label'])
        
        return InputExample(texts=[s1,s2], label=score)
    def __len__(self):
        return len(self.data)


path = './cail/sentence_relation_set.json'
num_epochs = 4
model_name = 'law_bert'
model_save_path = 'output/-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
trainset = downstramset(data = path,sample=10000)
train_dataloader = DataLoader(trainset,shuffle=True)
train_loss  = losses.CosineSimilarityLoss(model=model)
def dict2input(item):
    s1 = item['s1']
    s2 = item['s2']
    score = str(item['label'])
    return  InputExample(texts=[s1,s2], label=score)
file = open(path,'r',encoding='utf-8')
dev_samples = []
test_samples = []
for i in range(16000):
    line = file.readline()
    if i>10000 and i<15000:
        dev_samples.append(json.loads(line))
    if i>=15000:
        test_samples.append(json.loads(line))
    


dev_samples = [dict2input(i) for  i in dev_samples]
test_samples = [dict2input(i) for i in test_samples]
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up


model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)