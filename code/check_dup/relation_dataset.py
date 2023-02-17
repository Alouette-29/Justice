import json
import numpy as np 
# resize cail dataset 
orignial_dataset = "./cail/competition_stage_2_train.json"
orignial_dataset = open(orignial_dataset,'r',encoding='utf-8')
cases = orignial_dataset.readlines()
cases = [json.loads(i) for i in cases]
processed_dataset = "./cail/relation_classification_stage_2_train.json"

left = "{"
right = "}\n"
for sample in cases:
    # casea and caseb are the list of sentences of each case 
    casea = sample['Case_A']
    caseb = sample['Case_B']
    Arelations = sample['Case_A_rationales'] 
    Brelations = sample['Case_B_rationales'] 
    for i in range(len(casea)):
        if i in Arelations:
            label = 1
        else:
            label = 0

        with open(processed_dataset,'a',encoding='utf-8') as output:
            strings = f"{left}\"sentence\":\"{casea[i]}\",\"label\":\"{label}\"{right}"
            output.write(strings)
    for i in range(len(caseb)):
        if i in Brelations:
            label = 1
        else:
            label = 0
        with open(processed_dataset,'a',encoding='utf-8') as output:
            strings = f"{left}\"sentence\":\"{caseb[i]}\",\"label\":\"{label}\"{right}"
            output.write(strings)


with open(processed_dataset,'r',encoding='utf-8') as file:
    lines = file.readlines()

final_dataset = open("./cail/relation_class_stage_2_train.json",'w',encoding='utf-8')
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



