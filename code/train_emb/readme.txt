这个文件夹放从头开始训练的代码
训练模型： bert+pooling = sentence_transformer 
参考： official documentation of sentence_transformer 
网页链接： https://www.sbert.net/
训练原因： 原有的sentence_transformer 嵌入效果不太好
基础模型： bert-base-chinese
方案一： 利用有监督数据进行训练。监督数据需要自己构造。 
数据集结构: (sentencea,sentenceb,similarity) 
损失函数: cosine similarity loss function 

方案二：无监督训练，可以尝试的模型有： masked language model etc.