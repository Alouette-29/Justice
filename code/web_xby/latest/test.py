#encoding:utf-8
import torch
from sentence_transformers import SentenceTransformer


def test_model(inputs:str,embedding_model,classifier):
    embedding = embedding_model.encode(inputs)
    embedding = torch.from_numpy(embedding)
    output = classifier(embedding)
    output = torch.argmax(output,dim=0)
    return output 

if __name__=='__main__':
    inputs = "河南省郑州航空港经济综合实验区人民法院执行裁定书（2020）豫0192执2437号之一申请执行人：谷巧平，女，汉族，1982年5月24日出生，住河南省新郑市。被执行人：谷小利，女，汉族，1981年5月17日出生，住河南省新郑市。关于谷巧平申请执行谷小利民间借贷纠纷一案，依据本院已经发生法律效力的（2020）豫0192民初3061号民事判决书，被执行人应向申请执行人偿还借款400，000元及利息。被执行人未履行生效法律文书确认的义务，申请执行人于2020年11月30日向本院申请执行。本院在执行过程中，实施以下执行行为：一、2020年11月30日，向被执行人发出执行通知书，责令其限期履行法律文书所确定的义务，但被执行人至今未履行义务。二、2020年12月3日，2021年1月14号，通过执行网络查控系统向金融机构、证券机构、网络支付机构、自然资源部等发出查询通知，查询被执行人名下财产，未发现可供执行的财产。三、2021年1月5日，前往郑州航空港经济综合实验区不动产登记中心调查，未发现被执行人不动产登记信息。四、2021年1月6日，前往郑州市不动产登记中心调查，未发现被执行人不动产登记信息。五、2021年1月8日，前往郑州住房公积金管理中心郑州航空港管理部调查，未发现被执行人缴存记录。六、因被执行人未履行生效法律文书确认的义务，本院于2021年1月11日决定对被执行人采取限制高消费措施，并纳入失信被执行人名单。七、2021年1月20日，前往被执行人住所地执行，未发现可供执行的财产。本院已告知申请执行人本案的执行情况、财产调查措施、被执行人的财产情况、终结本次执行程序的依据及法律后果等，申请执行人在指定期限内不能向本院提供被执行人的可供执行的财产线索。本院认为，本案未发现被执行人可供执行财产。依照《最高人民法院关于适用<中华人民共和国民事诉讼法>的解释》第五百一十九条规定，裁定如下：终结本次执行程序。被执行人负有继续向申请执行人履行债务的义务，被执行人自动履行完毕的，当事人应当及时告知本院。申请执行人发现被执行人有可供执行财产的，可向本院或其他有管辖权的法院申请恢复执行，申请执行人申请恢复执行不受申请执行时效期间的限制。本裁定送达后立即生效。不服本裁定的，可以在收到本裁定之日起六十日内，依照《中华人民共和国民事诉讼法》第二百二十五条向本院提出执行异议。审判长崔纪国审判员田野审判员周勇增二〇二一年一月二十五日法官助理刘晓露书记员张艳涛"
    modeltypes = ['double']
    embedding_model = SentenceTransformer("./Representation/sim12w")
    classifier = torch.load("./Representation/sim12w_double_100th.model", map_location=torch.device('cpu'))
    # python test.py --modelname tsade15w --ckptepoch 10 
    out  = test_model(inputs,embedding_model,classifier)
    print(out)
