import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE 
from sklearn.cluster import KMeans
import torch
import numpy as np
import pandas as pd
from tqdm import trange 
def painting_without_label(title=str,x=None):
    x_std = StandardScaler().fit_transform(x)
    x_pca = PCA(n_components=2).fit_transform(x_std)
    df_pca = pd.DataFrame(x_pca,columns=['1st_Component','2nd_Component'])
    sns.scatterplot(data= df_pca,x='1st_Component',y='2nd_Component')
    plt.savefig(title+'pca.png')
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_std)
    df_tsne=pd.DataFrame(x_tsne,columns=['Dim1','Dim2'])
    plt.figure(figsize=(8,8))
    sns.scatterplot(data=df_tsne,x='Dim1',y='Dim2')
    plt.savefig(title+'tnse.png')
    plt.show()

def painting_with_label(title=str,x=None,y=None):
    x_std = StandardScaler().fit_transform(x)
    x_pca = PCA(n_components=2).fit_transform(x_std)
    x_pca =np.vstack((x_pca.T ,y)).T
    df_pca = pd.DataFrame(x_pca,columns=['1st_Component','2nd_Component','class'])
    plt.figure(figsize=(8,8))
    sns.scatterplot(data= df_pca,hue = 'class',x='1st_Component',y='2nd_Component')
    plt.savefig(str(title)+'_pca.png')
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_std)
    print(x_tsne.shape)
    x_tsne =np.vstack((x_tsne.T ,y)).T
    print(x_tsne.shape)
    df_tsne=pd.DataFrame(x_tsne,columns=['Dim1','Dim2','class'])
    plt.figure(figsize=(8,8))
    sns.scatterplot(data=df_tsne,hue ='class' , x='Dim1',y='Dim2')
    plt.savefig(str(title)+'_tsne.png')

def k_SSE(title,X,clusters): 
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
    plt.savefig(title+".png")
    plt.show()

if __name__=='__main__':
    path = "./ensence_tensor.pth"
    tensor = np.array(torch.load(path).to("cpu"))
    array1 = np.load("./subset/emb_tech_passage.npy")
    array2 = np.load("./subset/emb_financial_passage.npy")
    #painting_without_label("selected_essence_tensor",tensor)
    X = np.vstack([tensor,array1,array2])
    y  = pd.DataFrame(["fake"]*len(tensor)+["tech"]*len(array1)+["finan"]*len(array2)).astype("object")
    print(X.shape,y.shape)
    painting_with_label("摆烂了，毁灭吧",X,y)
    k_SSE("selected_essence_tensor_SSE",tensor,15)
