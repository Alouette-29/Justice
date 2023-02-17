from sentence_transformers import SentenceTransformer
import numpy as np 
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
with open("./成都高新川商小额贷款有限公司.txt",'r',encoding='utf-8') as f:
    cases = f.readlines()
cases = [i[200:300] for i  in cases]
embeddings = model.encode(cases)

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange 
np.save("loancompany.npy",embeddings)
#构造自定义函数，用于绘制不同的k值和对应总的簇内离差平方和的折线图
def k_SSE(X,clusters): 
    #选择连续的K种不同的值 
    K= trange(1,clusters+1) 
    #构建空列表用于存储总的簇内离差平方和 
    TSSE= [] 
    for k in K: 
    #用于存储各个簇内离差平方和 
        SSE = [] 
        kmeans = KMeans(n_clusters=k) 
        kmeans.fit(X) 
        #返回簇标签 
        labels = kmeans.labels_ 
        #返回簇中心 
        centers = kmeans.cluster_centers_ 
        #计算各簇样本的离差平方和，并保存到列表中 
        for label in set(labels): 
            SSE.append(np.sum((X.loc[labels==label,]-centers[label,:])**2)) 
            #计算总的簇内离差平方和 
        TSSE.append(np.sum(SSE)) #中文和负号正常显示 
    plt.rcParams['font.sans-serif'] = 'SimHei' 
    plt.rcParams['axes.unicode_minus'] =False  
    #设置绘画风格 
    plt.style.use('ggplot') 
    #绘制K的个数与TSSE的关系 
    plt.plot([i for i in range(15)],TSSE,'-') 
    plt.xlabel('簇的个数') 
    plt.ylabel('簇内离差平方和之和') 
    plt.show()
df = pd.DataFrame(embeddings)
k_SSE(df,15)

