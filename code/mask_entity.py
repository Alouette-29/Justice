#希望用实体识别工具将人名替换成甲乙丙丁 
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
pretrained = AutoModel.from_pretrained('hfl/rbt6')
from NER_model import NER_model as Model 
model = Model()
tokenizer = AutoTokenizer.from_pretrained('hfl/rbt6')

class NER_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuneing = False
        self.pretrained = None

        self.rnn = torch.nn.GRU(768, 768,batch_first=True)
        self.fc = torch.nn.Linear(768, 8)

    def forward(self, inputs):
        #print(inputs.keys())
        if self.tuneing:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(**inputs).last_hidden_state

        out, _ = self.rnn(out)

        out = self.fc(out).softmax(dim=2)

        return out

    def fine_tuneing(self, tuneing):
        self.tuneing = tuneing
        if tuneing:
            for i in pretrained.parameters():
                i.requires_grad = True

            pretrained.train()
            self.pretrained = pretrained
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)

            pretrained.eval()
            self.pretrained = None
def extract_entity(input_id,tag):
    #输出tag
    location = []
    name =  []
    organ = []
    for j in range(len(tag)):
        if tag[j] == 0 or tag[j] == 7:
            continue
        if tag[j] == 1:
            s = ''
            while(tag[j]==2 or tag[j]==1):
                s += tokenizer.decode(input_id[j])
                j+=1
            name.append(s)
        if tag[j] == 3:
            s = ''
            while(tag[j]==4 or tag[j]==3):
                s += tokenizer.decode(input_id[j])
                j+=1
            organ.append(s)
        if tag[j] == 5:
            s = ''
            while(tag[j]==5 or tag[j]==6):
                s += tokenizer.decode(input_id[j])
                j+=1
            location.append(s)
    return name,organ,location

import re 
maskname = ["甲","乙","丙","丁"]
maskorgan = ['a','b','c','d','e','f']
def deduction_mask(input):
    model = torch.load('D:/GitHub/NER_in_Chinese/model/命名实体识别_中文.model')
    model.eval()
    sentences = []
    for i in range(len(input)):
        sentences.append([word for word in input[i]])

    token = tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=137,
        is_split_into_words=True,
        return_tensors='pt',
        pad_to_multiple_of=137
        )

    with torch.no_grad():
        #[b, lens] -> [b, lens, 8] -> [b, lens]
        outs = model(token).argmax(dim=2)
    for i in range(len(sentences)):
        #移除pad
        select = token['attention_mask'][i] == 1
        input_id = token['input_ids'][i, select]
        out = outs[i, select]
        
        name ,organ,location = extract_entity(input_id,out)
        #print('==========================')
        #return mask_dict
        #这个地方或许可以直接匹配，或者用个map之类的
        
        for n in range(len(name)) :
            input[i] =  re.sub(pattern,maskname[n],input[i])
        for n in range(len(organ)) :
            pattern = re.compile(organ[n])
            input[i] =  re.sub(pattern,maskorgan[n],input[i])
    return input



###一个小小的试验 关于embedding的
text1    = '在执行过程中，本院依法采取了下列措施：\n一、已向被执行人张三、李四送达了执行通知书、财产报告令、执行催告通知书，责令其在三日内履行生效法律文书确定的义务，但被执行人至今未履行，亦未向本院申报财产。'
text2    = '在执行过程中，本院依法采取了下列措施：\n一、已向被执行人马克、陆静送达了执行通知书、财产报告令、执行催告通知书，责令其在三日内履行生效法律文书确定的义务，但被执行人至今未履行，亦未向本院申报财产。'
text3    = '在执行过程中，法院依法采取了下列措施：\n一、已向被执行人马克、陆静送达了执行通知书、财产报告令、执行催告通知书，责令其在三日内履行生效法律文书确定的义务，但被执行人至今未履行，亦未向法院申报财产。'
textbase = '在执行过程中，本院依法采取了下列措施：\n一、已向被执行人甲甲、乙乙送达了执行通知书、财产报告令、执行催告通知书，责令其在三日内履行生效法律文书确定的义务，但被执行人至今未履行，亦未向本院申报财产。'

example = [text1,text2,text3,textbase]
for i in range(len(example)):
    example[i] = [word for word in example[i]]
token = tokenizer.batch_encode_plus(
        example,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=137,
        is_split_into_words=True,
        return_tensors='pt',
        pad_to_multiple_of=137
        )

out = pretrained(**token).last_hidden_state

print(torch.mean(torch.cosine_similarity(out[0],out[1])))
print(torch.mean(torch.cosine_similarity(out[2],out[1])))
print(torch.mean(torch.cosine_similarity(out[0],out[2])))
print(torch.mean(torch.cosine_similarity(out[0],out[3])))
print(torch.mean(torch.cosine_similarity(out[1],out[3])))
print(torch.mean(torch.cosine_similarity(out[2],out[3])))
print(torch.mean(torch.cosine_similarity(out[3],out[3])))

print(torch.sum((out[0]-out[1])**2))
print(torch.sum((out[1]-out[2])**2))
print(torch.sum((out[2]-out[0])**2))
print(torch.sum((out[3]-out[0])**2))
print(torch.sum((out[3]-out[1])**2))
print(torch.sum((out[3]-out[2])**2))
print(torch.sum((out[3]-out[3])**2))

####result in my device
# tensor(0.9844, grad_fn=<MeanBackward0>)
# tensor(0.9886, grad_fn=<MeanBackward0>)
# tensor(0.9732, grad_fn=<MeanBackward0>)
# tensor(0.9807, grad_fn=<MeanBackward0>)
# tensor(0.9794, grad_fn=<MeanBackward0>)
# tensor(0.9677, grad_fn=<MeanBackward0>)
# tensor(1., grad_fn=<MeanBackward0>)
# tensor(891.6225, grad_fn=<SumBackward0>)
# tensor(710.3116, grad_fn=<SumBackward0>)
# tensor(1590.7209, grad_fn=<SumBackward0>)
# tensor(1136.7180, grad_fn=<SumBackward0>)
# tensor(1203.5551, grad_fn=<SumBackward0>)
# tensor(1924.1237, grad_fn=<SumBackward0>)
# tensor(0., grad_fn=<SumBackward0>)