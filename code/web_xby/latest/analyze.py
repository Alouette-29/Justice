import torch
import jieba
import json
import dill as pickle
import numpy as np
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from FastTextRank.FastTextRank4Sentence import FastTextRank4Sentence


class Posting():
    special_docid = -1
    def __init__(self, docid, tf=0):
        self.docid = docid
        self.tf = tf
    

def get_postings_list(inv_index, query_term):   
    try:
        return inv_index[query_term][1:]
    except KeyError:
        return []


def test_model(inputs:str,embedding_model,classifier):
    embedding = embedding_model.encode(inputs)
    embedding = torch.from_numpy(embedding)
    output = classifier(embedding)
    output = torch.argmax(output,dim=0)
    return output 


def qualified(term):
    with open('./data/baidu_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    for c in term:
        if c < '\u4e00' or c > '\u9fa5':
            return False
    if len(term.strip()) <= 1:
        return False
    if term in stopwords:
        return False
    return True


def cos_scores(query, k, IDF, inv_index, doc_length):
    log_func = lambda x: 1.0 + np.log10(x) if x > 0 else 0.0
    scores = defaultdict(lambda: 0.0)  
    query_terms = Counter(term for term in jieba.cut(query, cut_all=True) if qualified(term))
    for q in query_terms:
        w_tq = log_func(query_terms[q]) * IDF[q]
        postings_list = get_postings_list(inv_index, q)
        for posting in postings_list:
            w_td = log_func(posting.tf) * IDF[q]
            scores[posting.docid] += w_td * w_tq

    results = [(docid, score / doc_length[docid]) for docid, score in scores.items()]
    results.sort(key=lambda x: -x[1])
    
    return results[0:k]


def find_sim(content, k):
    with open('./data/fake_set.txt', 'r', encoding='utf-8') as f:
        cases = f.readlines()
    with open('./savings_fake/inv_index.pkl', 'rb') as f:
        inv_index = pickle.load(f)

    N = len(cases)  # 总共有N个文档
    IDF = defaultdict(lambda : 0)  
    for term in inv_index:
        df = len(inv_index[term]) - 1  
        IDF[term] = np.log10(N / df) 
    with open('./savings_fake/doc_length.pkl', 'rb+') as f:
        doc_length = pickle.load(f)
    
    res = cos_scores(content, k, IDF, inv_index, doc_length)

    similar = []
    for docid, sim in res:
        case = json.loads(cases[docid])
        case_name = case['case_name']
        case_id = case['case_id']
        simcase = {"name" : "案件名称：" + case_name, "id" : "案件ID：" + case_id, "sim" : "文本相似度：" + str(sim/100)}
        similar.append(simcase)
    
    return similar


def analysize(content):
    mod = FastTextRank4Sentence(use_w2v=False,tol=0.0001)
    summary = mod.summarize(content, 3)

    embedding_model = SentenceTransformer("./Representation/sim12w")
    classifier = torch.load("./Representation/sim12w_double_100th.model", map_location=torch.device('cpu'))
    out  = test_model(content, embedding_model, classifier)
    flag = out.item()

    if flag:
        classify = "此案件极有可能是虚假诉讼。"
    else:
        classify = "此案件可排除虚假诉讼嫌疑。"

    similar = find_sim(content, 3)  # 默认前3条相似的

    summary = summary[0] + summary[1] + summary[2]

    return classify, similar, summary


