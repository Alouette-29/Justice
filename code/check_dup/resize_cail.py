import json
import numpy as np 
# resize cail dataset 
orignial_dataset = "./cail/competition_stage_2_train.json"
orignial_dataset = open(orignial_dataset,'r',encoding='utf-8')
cases = orignial_dataset.readlines()
cases = [json.loads(i) for i in cases]
processed_dataset = "./cail/sentence_level_stage_2_train.json"
num = 0
# puts them in sentences 
for sample in cases:
    # casea and caseb are the list of sentences of each case 
    casea = sample['Case_A']
    caseb = sample['Case_B']
    relations = sample['relation'] # 2d list consist of several list which contain of sentence
    for relation in relations:
        with open(processed_dataset,'a',encoding='utf-8') as output:
            sentencea = casea[relation[0]]
            sentenceb = caseb[relation[1]]
            left = '{'
            right = '}\n'
            strings = f'{left}\"s1\":\"{sentencea}\",\"s2\":\"{sentenceb}\",\"label\":\"1\"{right}'
            output.write(strings)
            strings = f'{left}\"s1\":\"{sentenceb}\",\"s2\":\"{sentencea}\",\"label\":\"1\"{right}'
            output.write(strings)
            num+=2

    # sample negative facts:
    # to make sure it will learn well, we take a cross resample 
    with open(processed_dataset,'a',encoding='utf-8') as output:
        pairs = len(relations)
        if pairs ==1 :
            continue
        for i in range(pairs):
            index = np.random.randint(low=0, high=pairs, size=2, dtype='l')
            if index[0]==index[1]:
                continue
            pair1 = relations[index[0]]
            pair2 = relations[index[1]]
            sentencea1 = casea[pair1[0]]
            sentenceb1 = caseb[pair1[1]]
            sentencea2 = casea[pair2[0]]
            sentenceb2 = caseb[pair2[1]]
            strings = f'{left}\"s1\":\"{sentencea1}\",\"s2\":\"{sentenceb2}\",\"label\":\"0\"{right}'
            output.write(strings)
            strings = f'{left}\"s1\":\"{sentencea2}\",\"s2\":\"{sentenceb1}\",\"label\":\"0\"{right}'
            output.write(strings)


# test json 
processed_dataset = open("./cail/sentence_level_stage_2_train.json",'r',encoding='utf-8')
final_dataset = open("./cail/final_stage_2_train.json",'w',encoding='utf-8')

lines = processed_dataset.readlines()
wrong_case = 0
positive = 0 
negative = 0 
for line in lines:
    try:
        cases = json.loads(line)
        if cases['label'] =='0':
            negative+=1
        else:
            positive+=1
        final_dataset.write(line)
    except:
        wrong_case+=1
        
if wrong_case == 0:
    print("successfully constructed a dataset")
    print(negative, "is negative", positive ,"is positive ")
else:
    print("fail to construct a datase, please check your code ")
    print(wrong_case, "cannont load")
    print(negative, "is negative", positive ,"is positive ")
