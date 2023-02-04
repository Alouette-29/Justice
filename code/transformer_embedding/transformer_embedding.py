import sys
sys.path.append('..')

# 这是我从hugging face 的网站找到预训练模型 
# 昨天经过我简单的对这个模型进行测试
# 发现他基本具有中文识别能力
from sentence_transformers import SentenceTransformer
import numpy as np

# 给出 读取案件数量，然后读取，嵌入，返回 
def embed(case_reader,sample:int):
    roundnum = sample//5
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = np.zeros(384).reshape([1,384])
    for i in range(roundnum):
        case1 = case_reader.get_dict()[1]['content']
        case2 = case_reader.get_dict()[1]['content']
        case3 = case_reader.get_dict()[1]['content'] 
        case4 = case_reader.get_dict()[1]['content'] 
        case5 = case_reader.get_dict()[1]['content']
        cases = [case1,case2,case3,case4,case5]
        tempembedding  = model.encode(cases)
        #print(tempembedding.shape)
        embeddings = np.vstack([embeddings,tempembedding])
    for i in range(sample%5):
        case = case_reader.get_dict()[1]['content']
        tempembedding  = model.encode([case])
        embeddings = np.vstack([embeddings,tempembedding])
    embeddings = np.delete(embeddings,0,axis=0)
    return embeddings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE 
# 画图 
# pca 和tnse 降维 
def painting(rounds=int,x=None):
    x_std = StandardScaler().fit_transform(x)
    x_pca = PCA(n_components=2).fit_transform(x_std)
    df_pca = pd.DataFrame(x_pca,columns=['1st_Component','2nd_Component'])
    #plt.figure(figsize=(8,8))
    sns.scatterplot(data= df_pca,x='1st_Component',y='2nd_Component')
    #plt.show()
    plt.savefig(str(rounds)+'round_pca.png')
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_std)
    df_tsne=pd.DataFrame(x_tsne,columns=['Dim1','Dim2'])
    df_tsne.head
    plt.figure(figsize=(8,8))
    sns.scatterplot(data=df_tsne,x='Dim1',y='Dim2')
    plt.savefig(str(rounds)+'round_tnse.png')

# 目的是 看看这embdding 的效果怎么样
# 从图上看起来还是不太行
# 初步的解决方案是：
# 重写文书 ， 降低噪音
# 提取关系，用图模型建模 

if __name__ =='__main__':
    from utensil.get_text import Case
    case_reader = Case('../p2-1-2021.txt',encoding='utf-8')
    for i in range(20):
        embedding = embed(case_reader,1000)
        print(embedding.shape)
        painting(i,embedding)