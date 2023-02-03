import torch
from transformers import AutoTokenizer
from transformers import AutoModel
import argparse
import os 
#定义下游模型
pretrained = AutoModel.from_pretrained('hfl/rbt6')
class Model(torch.nn.Module):
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


model = Model()
def deduction(sentences,output_file):
    
    tokenizer = AutoTokenizer.from_pretrained('hfl/rbt6')
    model = torch.load('D:/GitHub/NER_in_Chinese/model/命名实体识别_中文.model')
    model.eval()
    for i in range(len(sentences)):
        sentences[i] = [word for word in sentences[i]]
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

    for i in range(1):
        #移除pad
        select = token['attention_mask'][i] == 1
        input_id = token['input_ids'][i, select]
        out = outs[i, select]
        
        #输出原句子
        print(tokenizer.decode(input_id).replace(' ', ''))
                #输出tag
        for tag in [out]:
            s = ''
            for j in range(len(tag)):
                if tag[j] == 0:
                    s += '·'
                    continue
                s += tokenizer.decode(input_id[j])
                s += str(tag[j].item())

            print(s)
        print('==========================')
    if output_file!=None:
        with open(output_file,'r') as f:
            f.write(s)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence",type = str,default="请你输入一个句子，作为我们识别的对象")
    parser.add_argument("--use_file",type =bool,default=False)
    parser.add_argument("--input_file",type =str,default=None)
    parser.add_argument("--output_file",type =str,default=None)
    return parser.parse_args()

arg = parse()
if(arg.use_file):
    filepath = arg.input_file
    if os.path.exists(filepath):
        with open(filepath,'r') as f:
            sentences = f.readlines()
else:
    sentences = [arg.sentence]
output_file =arg.output_file
sentences = ["请你输入一个句子，作为我们识别的对象"]
output_file = None
deduction(sentences=sentences,output_file = output_file)